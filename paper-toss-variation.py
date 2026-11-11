import argparse
import numpy as np
import cvxpy as cp
import time
from level import load_level
from visualization.viz import visualize

################################################################
# CLI
################################################################
parser = argparse.ArgumentParser(description='Paper-toss MPC solver')
parser.add_argument('level', help='Path to a level TOML file (e.g. levels/level1.toml)')
args = parser.parse_args()

################################################################
# Load level
################################################################
lv = load_level(args.level)

N         = lv.N
H         = lv.H
dt        = lv.dt
ALPHA_MAX = lv.ALPHA_MAX
r0        = lv.r0
v0        = lv.v0
r_goal    = lv.r_goal
fans      = lv.fans
N_FANS    = len(fans)

g = np.array([0.0, 0.0, -9.8])

################################################################
# Utils
################################################################

# returns the list of positions for `steps` time steps, representing the path taken via free
# flight given the initial position r0 and initial velocity v0
def simulate_free_flight(r0, v0, steps):
    r = np.zeros((steps, 3))
    v = np.zeros((steps, 3))
    r[0], v[0] = r0, v0
    for k in range(steps - 1):
        r[k+1] = r[k] + v[k] * dt + 0.5 * g * dt ** 2
        v[k+1] = v[k] + g * dt
    return r

r_free = simulate_free_flight(r0, v0, N)

################################################################
# MPC Setup
################################################################

# ARCHITECTURE NOTE: Compile-Once, Update-Often MPC Architecture (DPP Compliant)
# - Auxiliary state variables (r_var, v_var) decouple the trajectory calculations 
#   for sparse matrix representation
# - Parameterized model (active_fans, horizon_selector) to avoid re-compilation with DPP caching,
#   dropping solve times from 45ms to ~1ms

# to kick things, we start by guessing our trajectory to be the straight line between the start
# position and the end goal position
r_pred = np.linspace(r0, r_goal, N)

r_actual   = [r0]
v_actual   = [v0]
alpha_log  = []
active_log = []
pred_trajs = []
step_times = []

# Parameter setup
# r_k and v_k are initial position and velocity
# active_fans[j, i] is a mask that marks when fan i has influence on the ball in horizon timestep j
# horizon_selector is a one-hot vector that marks the last horizon timestep
r_k = cp.Parameter(3)
v_k = cp.Parameter(3)
active_fans = cp.Parameter((H, N_FANS))
horizon_selector = cp.Parameter(H + 1)

# Variable setup
# r_var and v_var are auxiliary variables representing ball positions and velocities at each timestep within a horizon
# alpha_var[j, i] represents the activation strength of fan i at horizon timestep j
r_var     = cp.Variable((H + 1, 3))
v_var     = cp.Variable((H + 1, 3))
alpha_var = cp.Variable((H, N_FANS))

# Constraints
# match initial conditions at each horizon timestep and bounded fan activation strengths
cons = [r_var[0] == r_k, v_var[0] == v_k, alpha_var <= ALPHA_MAX, alpha_var >= -ALPHA_MAX]
for j in range(H):
    wind_j = sum(alpha_var[j, i] * active_fans[j, i] * fans[i].direction
                    for i in range(N_FANS))
    acc_j = g + wind_j
    cons += [
        r_var[j + 1] == r_var[j] + v_var[j] * dt + 0.5 * acc_j * dt ** 2,
        v_var[j + 1] == v_var[j] + acc_j * dt
    ]

# Use a selector to dynamically modify horizon length in objective (ensuring DPP compliance)
horizon_end = sum(horizon_selector[j] * r_var[j] for j in range(H+1))
obj = cp.Minimize(cp.sum_squares(horizon_end - r_goal))
prob = cp.Problem(obj, cons)

# Dummy solve to trigger compilation before MPC Loop
t0 = time.monotonic()

r_k.value, v_k.value = np.zeros(3), np.zeros(3)
active_fans.value, horizon_selector.value = np.zeros((H, N_FANS)), np.zeros(H + 1)
prob.solve(solver=cp.CLARABEL, verbose=False)

step_time = (time.monotonic() - t0) * 1000
print(f"Dummy solve time: {step_time:.2f} ms")

################################################################
# Main MPC Loop
################################################################

print(f"MPC  fixed horizon H={H}  (N={N}, dt={dt}s, T={(N-1)*dt:.1f}s)")
print(f"  {'k':>3}  {'horizon':>7}  {'actual err':>12}  {'status':>10}  {'solve time':>8}")
print("  " + "─" * 56)

for k in range(N - 1):
    t0 = time.monotonic()

    horizon = min(H, N - k - 1)
    horizon_mask = np.zeros(H+1)
    horizon_mask[horizon] = 1
    horizon_selector.value = horizon_mask

    r_k.value = r_actual[k]
    v_k.value = v_actual[k]

    # Determine `active` fans (i.e. those that have an influence over the next `horizon` timesteps)
    active = np.zeros((H, N_FANS))
    r_check = r_pred[k:k+horizon]
    for i, fan in enumerate(fans):
        active_horizon = np.zeros((H,))
        active_horizon[:horizon] = fan.contains(r_check).astype('float')
        active[:, i] = active_horizon

    active_fans.value = active

    prob.solve(solver=cp.CLARABEL, verbose=False)

    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        print(f"  {k:>3}  {horizon:>7}  {'—':>12}  {status:>10}  {(time.time() - t0)*1000:>8.2f}ms  ← FAILED")
        break
    assert r_var.value is not None
    assert v_var.value is not None
    assert alpha_var.value is not None

    # Update r_pred using the steps in the computed horizon
    horizon_preds = r_var.value[:horizon+1]
    r_pred[k:k+horizon+1] = horizon_preds

    # Extract first action
    alpha_k  = alpha_var.value[0]
    active_k = active[0]

    alpha_log.append(alpha_k)
    active_log.append(active_k)
    pred_trajs.append(horizon_preds)

    # Update actual state with result of first action
    r_next = r_var.value[1]
    v_next = v_var.value[1]

    r_actual.append(r_next)
    v_actual.append(v_next)

    err = float(np.linalg.norm(r_next - r_goal))
    step_time = (time.monotonic() - t0) * 1000
    step_times.append(step_time)
    print(f"  {k:>3}  {horizon:>7}  {err:>12.4f}  {status:>10}  {step_time:>8.2f}ms")

r_actual   = np.array(r_actual)
alpha_log  = np.array(alpha_log)
active_log = np.array(active_log)
final_err = float(np.linalg.norm(r_actual[-1] - r_goal))
total_time = sum(step_times)
avg_time = np.mean(step_times)
print(f"\nActual terminal error: {final_err:.4f} m")
print(f"Total time: {total_time:.2f} ms, average per-step time: {avg_time:.2f} ms")


################################################################
# Visualization!
################################################################

visualize(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
          final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal)
