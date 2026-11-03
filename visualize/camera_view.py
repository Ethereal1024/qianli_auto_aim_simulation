import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from .view import View
import numpy as np


class CameraView(View):
    def __init__(
        self,
        ax,
        fx: float = 800.0,
        img_height: int = 1024,
        img_width: int = 1280,
        armor_colors: tuple[str, ...] = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd"),
    ):
        super().__init__(ax)
        self.n_armors = len(armor_colors)

        self.fills, self.edges, self.tags = [], [], []
        for m, color in enumerate(armor_colors):
            fill = Polygon(
                np.zeros((4, 2)), closed=True, lw=0, alpha=0.35, facecolor=color
            )
            ax.add_patch(fill)
            self.fills.append(fill)
            self.edges.append(ax.plot([], [], color=color, lw=1.2)[0])
            self.tags.append(ax.text(0, 0, f"A{m}", fontsize=8))
