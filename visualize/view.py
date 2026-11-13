from abc import ABC, abstractmethod

import numpy as np


class View(ABC):
    def __init__(self, ax, timeline: np.ndarray):
        self.ax = ax
        self.timeline = timeline
        self._i = 0
        
    @abstractmethod
    def update(self, t: float) -> bool: ...
    
    @abstractmethod
    def get_moving_items(self) -> list: ...
    
    def _closed(self, points):
        return np.vstack([points, points[:1]])
    
    def _time_query(self, t: float, timeline: np.ndarray) -> int:
        n = len(timeline)
        i = self._i
        while i + 1 < n and abs(timeline[i + 1] - t) < abs(timeline[i] - t):
            i += 1
        self._i = i
        return i