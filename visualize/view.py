from abc import ABC, abstractmethod


class View(ABC):
    def __init__(self, ax):
        self.ax = ax
        
    @abstractmethod
    def update(self, index: int) -> None: ...