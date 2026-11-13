from collections import deque

import numpy as np
from matplotlib.patches import Polygon

from tools.math import geometry as gmt

from .view import View


class CameraView(View):
    def __init__(
        self,
        ax,
        timeline: np.ndarray,
        armor_corners: np.ndarray,
        armor_centers: np.ndarray,
        cam_rots: np.ndarray,
        cam_trans: np.ndarray,
        hit_positions: np.ndarray | None = None,
        t_fly: np.ndarray | None = None,
        fx: float = 4000.0,
        img_height: int = 1024,
        img_width: int = 1280,
        armor_colors: tuple[str, ...] = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd"),
    ):
        super().__init__(ax, timeline)
        self.n_armors = len(armor_colors)

        self.armor_corners = armor_corners
        self.armor_centers = armor_centers
        self.rots_wc = cam_rots
        self.trans_wc = cam_trans

        self.fx = fx
        self.img_height = img_height
        self.img_width = img_width

        self.ax.set_xlim(0, img_width)
        self.ax.set_ylim(img_height, 0)
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.3)
        self.ax.set_autoscale_on(False)
        self.ax.set_title("Camera View")

        self.fills, self.edges, self.tags = [], [], []
        for m, color in enumerate(armor_colors):
            fill = Polygon(
                np.zeros((4, 2)), closed=True, lw=0, alpha=0.35, facecolor=color
            )
            ax.add_patch(fill)
            self.fills.append(fill)
            self.edges.append(ax.plot([], [], color=color, lw=1.2)[0])
            self.tags.append(ax.text(0, 0, f"A{m}", fontsize=8))

        self.hit_positions = hit_positions
        self.hit_timeline = timeline + t_fly if t_fly is not None else None

        self.fade_duration = 0.01
        self.hit_scatter = ax.scatter(
            [], [], marker="x", s=100,
            linewidths=2, c="black",
        )
        self.hit_scatter.set_visible(False)

    def update(self, t: float):
        if t > self.timeline[-1]:
            return False
        
        index = self._time_query(t, self.timeline)
        armor_kps = self.armor_corners[index]
        armor_centers = self.armor_centers[index]
        r_wc, t_wc = self.rots_wc[index], self.trans_wc[index]
        r_cw, t_cw = gmt.invert_transform(r_wc, t_wc)

        if self.hit_positions is not None and self.hit_timeline is not None:
            ages = t - self.hit_timeline
            active = (ages >= 0) & (ages < self.fade_duration)
            active_idx = np.where(active)[0]
            
            if len(active_idx) > 0:
                positions = self.hit_positions[active_idx]
                u_hits, v_hits = self._camera_project(positions, r_cw, t_cw)
                alphas = np.clip(1.0 - ages[active_idx] / self.fade_duration, 0, 1)
                # print(f"t={t:.2f}, K={len(active_idx)}, u={u_hits[:3]}, v={v_hits[:3]}")

                rgba = np.zeros((len(active_idx), 4))
                rgba[:, :3] = 0.0
                rgba[:, 3] = alphas
                
                self.hit_scatter.set_offsets(np.column_stack([u_hits, v_hits]))
                self.hit_scatter.set_facecolors(rgba)
                self.hit_scatter.set_visible(True)
            else:
                self.hit_scatter.set_visible(False)
                    
                    

        u, v = self._camera_project(armor_kps, r_cw, t_cw)  # (armors, 4)
        normal = np.cross(
            armor_kps[:, 1] - armor_kps[:, 0], armor_kps[:, 3] - armor_kps[:, 0]
        )
        front = (normal * (t_wc - armor_centers)).sum(-1) > 0
        visible = self._quad_in_view(u, v)
        for m in range(self.n_armors):
            if not visible:
                self.fills[m].set_visible(False)
                self.edges[m].set_visible(False)
                self.tags[m].set_visible(False)
            else:
                quad = self._closed(np.column_stack([u[m], v[m]]))
                self.fills[m].set_xy(quad)
                self.fills[m].set_visible(front[m])

                self.edges[m].set_visible(True)
                self.edges[m].set_data(quad[:, 0], quad[:, 1])
                self.edges[m].set_linestyle("-" if front[m] else ":")

                self.tags[m].set_visible(True)
                self.tags[m].set_position((u[m].mean(), v[m].mean()))
        return True

    def get_moving_items(self) -> list:
        return [*self.fills, *self.edges, *self.tags, self.hit_scatter]

    def _camera_project(
        self, points_world: np.ndarray, r_cw: np.ndarray, t_cw: np.ndarray
    ):
        points_cam = gmt.apply_transform(r_cw, t_cw, points_world)
        points_img = gmt.project_to_normalized(points_cam)
        u = self.fx * points_img[..., 0] + self.img_width / 2
        v = self.fx * points_img[..., 1] + self.img_height / 2
        return u, v

    def _quad_in_view(self, u, v):
        return (
            v.max() > 0
            and v.min() < self.img_height
            and u.max() > 0
            and u.min() < self.img_width
        )
