import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

OUT_DIR = os.path.dirname(__file__)
_FIELD_GRID = 3   # arrows per axis in force-field mode


def _box_faces(bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    return [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(0 if False else x0,y1,z0) if False else (x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
    ]


class Visualizer:
    def __init__(self, fans, r_free, r_actual, alpha_log, active_log,
                 pred_trajs, final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal):
        self.fans       = fans
        self.r_free     = r_free
        self.r_actual   = r_actual
        self.alpha_log  = alpha_log
        self.active_log = active_log
        self.pred_trajs = pred_trajs
        self.final_err  = final_err
        self.H          = H
        self.N          = N
        self.dt         = dt
        self.ALPHA_MAX  = ALPHA_MAX
        self.r0         = r0
        self.v0         = v0
        self.r_goal     = r_goal
        self.eff_alpha  = active_log * alpha_log
        self.total_act  = self.eff_alpha.sum(axis=1)
        self.t          = np.arange(len(alpha_log)) * dt

    def _draw_fans(self, ax, alpha_box=0.10, force_field=False, grid_override=None):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        view_bounds = np.array([xlim, ylim, zlim])
        view_spans = view_bounds[:, 1] - view_bounds[:, 0]
        mean_span = view_spans.mean()

        grid_density = grid_override if grid_override is not None else _FIELD_GRID

        base_len = mean_span * 0.08

        drawn = set()
        for fan in self.fans:
            clipped_low = np.maximum(fan.bounds[:, 0], view_bounds[:, 0])
            clipped_high = np.minimum(fan.bounds[:, 1], view_bounds[:, 1])
            
            if np.any(clipped_low >= clipped_high):
                continue

            clipped_bounds = np.stack([clipped_low, clipped_high], axis=1)
            key = tuple(clipped_bounds.ravel())

            if key not in drawn:
                ax.add_collection3d(Poly3DCollection(
                    _box_faces(clipped_bounds), alpha=alpha_box,
                    facecolor='#AAAAAA', edgecolor='#888888', linewidth=0.4,
                ))
                drawn.add(key)

            raw_dir = fan.direction
            norm_dir = raw_dir / (np.linalg.norm(raw_dir) + 1e-8)
            
            scaled_dir_components = norm_dir / (view_spans + 1e-8)
            scale_correction = np.linalg.norm(scaled_dir_components) * mean_span
            corrected_dir = norm_dir / (scale_correction + 1e-8)
            
            d = corrected_dir * base_len

            eff_span = abs(norm_dir[0]) * view_spans[0] + abs(norm_dir[1]) * view_spans[1] + abs(norm_dir[2]) * view_spans[2]
            adaptive_ratio = np.clip(0.12 * (mean_span / (eff_span + 1e-8)), 0.04, 0.25)

            if force_field:
                lo = clipped_bounds[:, 0] + np.maximum(0.0, -d) * 0.5
                hi = clipped_bounds[:, 1] - np.maximum(0.0,  d) * 0.5
                if np.any(lo >= hi):
                    continue
                xs, ys, zs = [np.linspace(lo[i], hi[i], max(3, grid_density)) for i in range(3)]
                gx, gy, gz = np.meshgrid(xs, ys, zs)
                ax.quiver(gx.ravel(), gy.ravel(), gz.ravel(),
                          np.full(gx.size, d[0]),
                          np.full(gx.size, d[1]),
                          np.full(gx.size, d[2]),
                          color=fan.color, linewidth=0.8,
                          arrow_length_ratio=adaptive_ratio, alpha=0.6)
            else:
                c = clipped_bounds.mean(axis=1)
                ax.quiver(c[0], c[1], c[2], d[0], d[1], d[2],
                          color=fan.color, linewidth=1.8,
                          arrow_length_ratio=adaptive_ratio, alpha=0.9)

    def _fig_overview(self):
        n_fans      = len(self.fans)
        ncols_right = max(1, int(np.ceil(np.sqrt(n_fans))))
        nrows_right = int(np.ceil(n_fans / ncols_right))

        fig = plt.figure(figsize=(16 + ncols_right * 3, max(8, nrows_right * 3.2)))
        
        gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[3.2, ncols_right], wspace=0.25)

        ax3d = fig.add_subplot(gs_outer[0, 0], projection='3d')

        gs_inner = gs_outer[0, 1].subgridspec(nrows_right, ncols_right, hspace=0.25, wspace=0.25)
        
        fan_axes = []
        for i in range(n_fans):
            r_idx = i // ncols_right
            c_idx = i % ncols_right
            fan_axes.append(fig.add_subplot(gs_inner[r_idx, c_idx]))

        ax3d.plot(*self.r_free.T, '--', color='#909090', linewidth=1.5, alpha=0.8)

        ax3d.plot(self.r_actual[:, 0], self.r_actual[:, 1], self.r_actual[:, 2],
                  color='steelblue', linewidth=3.0)

        ax3d.scatter(*self.r0,     s=140, c='lime',  marker='o', depthshade=False, zorder=6)
        ax3d.scatter(*self.r_goal, s=200, c='red',   marker='o', depthshade=False, zorder=6)
        ax3d.scatter(*self.r_goal, s=40,  c='white', marker='o', depthshade=False, zorder=7)

        all_pts = [self.r0, self.r_goal, self.r_actual]
        for traj in self.pred_trajs:
            all_pts.append(traj)
        all_pts = np.vstack(all_pts)

        min_xyz = all_pts.min(axis=0) - 0.3
        max_xyz = all_pts.max(axis=0) + 0.3

        ax3d.set_xlim(min_xyz[0], max_xyz[0])
        ax3d.set_ylim(min_xyz[1], max_xyz[1])
        ax3d.set_zlim(min_xyz[2], max_xyz[2])

        v_span = max_xyz - min_xyz
        v_norm = np.linalg.norm(self.v0)
        target_v_len = v_span.mean() * 0.15
        v_scale = target_v_len / v_norm if v_norm > 0 else 1.0
        v_scaled_vec = self.v0 * v_scale
        v_unit = v_scaled_vec / (np.linalg.norm(v_scaled_vec) + 1e-8)
        v_eff_span = abs(v_unit[0]) * v_span[0] + abs(v_unit[1]) * v_span[1] + abs(v_unit[2]) * v_span[2]
        v_arrow_ratio = np.clip(0.12 * (v_span.mean() / (v_eff_span + 1e-8)), 0.04, 0.25)

        ax3d.quiver(*self.r0, *v_scaled_vec,
                    color='black', linewidth=1.8, arrow_length_ratio=v_arrow_ratio)

        self._draw_fans(ax3d, force_field=True)

        legend_els = [
            Line2D([0],[0], color='#909090',   linestyle='--', label='Free flight'),
            Line2D([0],[0], color='steelblue', linewidth=3,    label='MPC trajectory'),
            Line2D([0],[0], color='black',     linewidth=1.8,  label='Initial velocity'),
        ] + [Line2D([0],[0], color=f.color, linewidth=6, alpha=0.55, label=f.name) for f in self.fans]
        ax3d.legend(handles=legend_els, fontsize=7, loc='upper left', framealpha=0.75)
        ax3d.set_xlabel('X (m)', labelpad=2)
        ax3d.set_ylabel('Y (m)', labelpad=2)
        ax3d.set_zlabel('Z (m)', labelpad=2)
        ax3d.view_init(elev=20, azim=-50)

        for i, (fan, ax) in enumerate(zip(self.fans, fan_axes)):
            act = self.eff_alpha[:, i]
            ax.fill_between(self.t, act, color=fan.color, alpha=0.35)
            ax.plot(self.t, act, color=fan.color, linewidth=1.8)
            ax.axhline(0, color='#999999', linewidth=0.6, linestyle='--')
            ax.set_ylim(-self.ALPHA_MAX * 1.1, self.ALPHA_MAX * 1.1)
            ax.set_title(fan.name, fontsize=8, color=fan.color, fontweight='bold', pad=3)
            ax.set_xlabel('t (s)', fontsize=7, labelpad=1)
            ax.set_ylabel('α (m/s²)', fontsize=7, labelpad=1)
            ax.tick_params(labelsize=6, pad=1)
            ax.grid(alpha=0.3)

        return fig

    def _fig_per_step(self):
        ncols = 5
        nrows = int(np.ceil((self.N - 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, subplot_kw={'projection': '3d'},
                                 figsize=(ncols * 4, nrows * 4))
        axes = np.array(axes).flatten()

        global_all_pts = [self.r0, self.r_goal, self.r_actual]
        for traj in self.pred_trajs:
            global_all_pts.append(traj)
        global_all_pts = np.vstack(global_all_pts)

        g_min_xyz = global_all_pts.min(axis=0) - 0.3
        g_max_xyz = global_all_pts.max(axis=0) + 0.3

        x_lims = (np.floor(g_min_xyz[0]), np.ceil(g_max_xyz[0]))
        y_lims = (np.floor(g_min_xyz[1]), np.ceil(g_max_xyz[1]))
        z_lims = (np.floor(g_min_xyz[2]), np.ceil(g_max_xyz[2]))

        for k, (traj, ax) in enumerate(zip(self.pred_trajs, axes)):
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_zlim(z_lims)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.zaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

            self._draw_fans(ax, alpha_box=0.07, force_field=True, grid_override=2)

            if k > 0:
                ax.plot(self.r_actual[:k+1, 0], self.r_actual[:k+1, 1],
                        self.r_actual[:k+1, 2], color='lime', linewidth=2.0)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color='steelblue', linewidth=1.8, linestyle='--')
            ax.scatter(*self.r0,          s=60,  c='lime',   marker='o', depthshade=False)
            ax.scatter(*self.r_actual[k], s=80,  c='yellow', marker='o', depthshade=False, zorder=6)
            ax.scatter(*self.r_goal,      s=120, c='red',    marker='o', depthshade=False)
            ax.set_title(f'k={k}  H={min(self.H, self.N - k)}', fontsize=9, fontweight='bold')
            ax.set_xlabel('X', fontsize=6, labelpad=-1)
            ax.set_ylabel('Y', fontsize=6, labelpad=-1)
            ax.set_zlabel('Z', fontsize=6, labelpad=-1)
            ax.tick_params(labelsize=5, pad=-2)
            ax.view_init(elev=20, azim=-50)

        for ax in axes[self.N-1:]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig

    def show(self):
        fig1 = self._fig_overview()
        fig3 = self._fig_per_step()

        plt.show()

        for fig, name in [(fig1, 'poc7_visualization.png'),
                          (fig3, 'poc7_per_step.png')]:
            fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
            print(f"Saved: {name}")


def visualize(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
              final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal):
    Visualizer(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
               final_err, H, N, dt, ALPHA_MAX, r0, v0, r_goal).show()