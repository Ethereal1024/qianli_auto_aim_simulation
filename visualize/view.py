from abc import ABC, abstractmethod

import numpy as np


class View(ABC):
    def __init__(self, ax):
        self.ax = ax
        
    @abstractmethod
    def update(self, index: int) -> None: ...
    
    @abstractmethod
    def get_moving_items(self) -> list: ...
    
    def _closed(self, points):
        return np.vstack([points, points[:1]])