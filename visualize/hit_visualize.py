import numpy as np
from matplotlib.colors import to_rgb


class HitVisualize:
    def __init__(
        self,
        hit_scatter,
        hit_timeline: np.ndarray,
        fire: np.ndarray,
        *,
        fade_duration: float = 0.1,
        color: str = "#6d52e5",
    ) -> None:
        self.hit_scatter = hit_scatter
        self.hit_timeline = hit_timeline
        self.fire = fire
        self.fade_duration = fade_duration
        self.rgb = np.array(to_rgb(color))

    def update(self, t: float, coordinates: list[np.ndarray]):
        ages = t - self.hit_timeline
        active = (ages >= 0) & (ages < self.fade_duration) & self.fire.squeeze(axis=-1)
        active_idx = np.where(active)[0]

        if len(active_idx) > 0:
            alphas = np.clip(1.0 - ages[active_idx] / self.fade_duration, 0, 1)
            rgba = np.zeros((len(active_idx), 4))
            rgba[:, :3] = self.rgb
            rgba[:, 3] = alphas

            self._set_offset(coordinates, active_idx)
            self.hit_scatter.set_facecolors(rgba)
            self.hit_scatter.set_visible(True)

        else:
            self.hit_scatter.set_visible(False)

    def _set_offset(self, coordinates: list[np.ndarray], active_idx: np.ndarray):
        if len(coordinates) == 2:
            x, y = coordinates[0][active_idx], coordinates[1][active_idx]
            self.hit_scatter.set_offsets(np.column_stack([x, y]))
        if len(coordinates) == 3:
            x, y, z = (
                coordinates[0][active_idx],
                coordinates[1][active_idx],
                coordinates[2][active_idx],
            )
            self.hit_scatter.set_offsets(np.column_stack([x, y]))
            self.hit_scatter.set_3d_properties(z, "z")
