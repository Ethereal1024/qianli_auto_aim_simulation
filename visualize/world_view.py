import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .view import View


class WorldView(View):
    def __init__(
        self,
        ax: Axes3D,
        armor_corners: np.ndarray,
        frames_metas: list[tuple[str, float]],
        frames_rots: list[np.ndarray],
        frames_trans: list[np.ndarray],
        title: str = "World View",
        axis_colors: tuple[str, ...] = ("#d62728", "#2ca02c", "#1f77b4"),
        armor_colors: tuple[str, ...] = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd"),
    ):
        super().__init__(ax)
        self.n_armors = len(armor_colors)
        self.armor_corners = armor_corners
        self.axis_colors = axis_colors
        self.armor_colors = armor_colors

        self.frames = [self._init_frame(*meta) for meta in frames_metas]
        self.frames_rots = frames_rots
        self.frames_trans = frames_trans

        self.armor_plates = [self._init_line3d(c, 1.2) for c in self.armor_colors]

        self._set_cube(
            np.vstack(
                [points.reshape(-1, 3) for points in [armor_corners, *frames_trans]]
            )
        )

        self.ax.set_title(title)
        self.ax.set_xlabel("X fwd (m)")
        self.ax.set_ylabel("Y left (m)")
        self.ax.set_zlabel("Z up (m)")

    def update(self, index: int) -> None:
        armor_kps = self.armor_corners[index]
        for frame, trans, rots in zip(self.frames, self.frames_trans, self.frames_rots):
            self._update_frame(frame, trans[index], rots[index])
        for m in range(self.n_armors):
            loop = self._closed(armor_kps[m]).T
            self.armor_plates[m].set_data_3d(*loop)

    def get_moving_items(self) -> list:
        return [
            a for lines, text, _ in self.frames for a in (*lines, text)
        ] + self.armor_plates

    def _set_cube(self, points):
        lo, hi = points.reshape(-1, 3).min(0), points.reshape(-1, 3).max(0)
        mid, half = (lo + hi) / 2, (hi - lo).max() / 2
        self.ax.set_xlim(mid[0] - half, mid[0] + half)
        self.ax.set_ylim(mid[1] - half, mid[1] + half)
        self.ax.set_zlim(mid[2] - half, mid[2] + half)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_autoscale_on(False)

    def _init_line3d(self, color, lw=2.0):
        line = self.ax.plot([], [], [], color=color, lw=lw)[0]
        return line

    def _init_text3d(self, label):
        text = self.ax.text(0, 0, 0, label, fontsize=8)
        return text

    def _init_frame(self, label, length):
        lines = [self._init_line3d(c) for c in self.axis_colors]
        return lines, self._init_text3d(label), length

    def _update_frame(self, frame, origin, rot):
        lines, label, length = frame
        for line, end in zip(lines, origin + length * rot.T):
            line.set_data_3d(*np.vstack([origin, end]).T)
        label.set_position_3d(origin)
