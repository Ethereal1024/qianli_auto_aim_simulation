import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .view import View


class Animator:
    def __init__(
        self,
        fig,
        views: list[View],
        fps: float
    ) -> None:
        self.fig = fig
        self.views = views
        self.fps = fps
        
        self.moving = [item for view in self.views for item in view.get_moving_items()]
        
    def play(self, seconds: float):
        n_frames = max(2, round(seconds * self.fps))
        self._update(0)
        start = time.perf_counter()
        
        def animate(_tick):
            t = time.perf_counter() - start
            self._update(t)
            if index == n_frames - 1 and animation.event_source is not None:
                animation.event_source.stop()
            return self.moving
                            
        animation = FuncAnimation(
            self.fig, 
            animate,
            interval=1000.0 / self.fps, 
            repeat=False, 
            blit=True, 
            cache_frame_data=False
        )
        
        plt.show()
        
    def _update(self, index):
        for view in self.views:
            view.update(index) 
