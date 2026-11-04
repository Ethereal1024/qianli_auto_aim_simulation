import numpy as np
from matplotlib.patches import Polygon

from tools.math import geometry as gmt

from .view import View


class CameraView(View):
    def __init__(
        self,
        ax,
        corners: np.ndarray,
        centers: np.ndarray,
        cam_rots: np.ndarray,
        cam_trans: np.ndarray,
        fx: float = 4000.0,
        img_height: int = 1024,
        img_width: int = 1280,
        armor_colors: tuple[str, ...] = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd"),
    ):
        super().__init__(ax)
        self.n_armors = len(armor_colors)

        self.corners = corners
        self.centers = centers
        self.rots_cw = cam_rots
        self.trans_cw = cam_trans

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

    def update(self, index: int):
        armor_kps = self.corners[index]
        body_center = self.centers[index]
        r_wc, t_wc = gmt.invert_transform(self.rots_cw[index], self.trans_cw[index])

        u, v = self._camera_project(armor_kps, r_wc, t_wc)  # (armors, 4)
        normal = np.cross(
            armor_kps[:, 1] - armor_kps[:, 0], armor_kps[:, 3] - armor_kps[:, 0]
        )
        front = (normal * (t_wc - body_center)).sum(-1) > 0
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

    def _camera_project(
        self, points_world: np.ndarray, r_wc: np.ndarray, t_wc: np.ndarray
    ):
        points_cam = gmt.apply_transform(r_wc, t_wc, points_world)
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
