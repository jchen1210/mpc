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
# Main MPC Loop
################################################################

# to kick things, we start by guessing our trajectory to be the straight line between the start
# position and the end goal position
r_pred = np.linspace(r0, r_goal, N)

r_actual   = [r0]
v_actual   = [v0]
alpha_log  = []
active_log = []
pred_trajs = []
step_times = []

print(f"MPC  fixed horizon H={H}  (N={N}, dt={dt}s, T={(N-1)*dt:.1f}s)")
print(f"  {'k':>3}  {'horizon':>7}  {'actual err':>12}  {'status':>10}  {'solve time':>8}")
print("  " + "─" * 56)

for k in range(N - 1):
    t0 = time.monotonic()

    r_k = r_actual[k]
    v_k = v_actual[k]

    horizon = min(H, N - k - 1)

    # determine `active` fans (i.e. those that have an influence over the next `horizon` timesteps)
    active = np.zeros((horizon, N_FANS))
    r_check = np.zeros((horizon, 3))
    r_check[0] = r_actual[k]
    r_check[1:] = r_pred[k+1:k+horizon]
    for i, fan in enumerate(fans):
        active[:, i] = fan.contains(r_check).astype('float')
    

    # now that we know which fans are active for each time step in this horizon, we can setup our problem
    # for the current iteration
    r_var     = cp.Variable((horizon + 1, 3))
    v_var     = cp.Variable((horizon + 1, 3))
    alpha_var = cp.Variable((horizon, N_FANS))

    # constraints!
    cons = [r_var[0] == r_k, v_var[0] == v_k]
    for j in range(horizon):
        wind_j = sum(active[j, i] * alpha_var[j, i] * fans[i].direction
                     for i in range(N_FANS))
        acc_j = g + wind_j
        cons += [
            r_var[j + 1] == r_var[j] + v_var[j] * dt + 0.5 * acc_j * dt ** 2,
            v_var[j + 1] == v_var[j] + acc_j * dt,
            alpha_var[j] <=  ALPHA_MAX,
            alpha_var[j] >= -ALPHA_MAX,
        ]

    obj = cp.Minimize(cp.sum_squares(r_var[-1] - r_goal))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        print(f"  {k:>3}  {horizon:>7}  {'—':>12}  {status:>10}  {(time.time() - t0)*1000:>8.2f}ms  ← FAILED")
        break

    # Update r_pred for the next H steps
    r_pred[k:k+horizon+1] = r_var.value

    # Extract first action
    alpha_k  = alpha_var.value[0]
    active_k = active[0]

    alpha_log.append(alpha_k)
    active_log.append(active_k)
    pred_trajs.append(r_var.value)

    # Simulate one true step
    wind_true = sum(active_k[i] * alpha_k[i] * fans[i].direction
                    for i in range(N_FANS))
    acc_true = g + wind_true
    r_next = r_k + v_k * dt + 0.5 * acc_true * dt ** 2
    v_next = v_k + acc_true * dt

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
'''
visualize(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
          final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal)
'''