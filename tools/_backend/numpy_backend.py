
from typing import Any

import numpy as np

from .base import ArrayBackend


class NumpyBackend(ArrayBackend):
    name = "numpy"

    def zeros_like(self, x):
        return np.zeros_like(x)

    def ones_like(self, x):
        return np.ones_like(x)

    def zeros(self, shape, *, dlike):
        return np.zeros(shape, dtype=dlike.dtype)

    def eye(self, n, *, dlike):
        return np.eye(n, dtype=dlike.dtype)

    def cos(self, x):
        return np.cos(x)
    
    def sin(self, x):
        return np.sin(x)
    
    def sqrt(self, x):
        return np.sqrt(x)
    
    def atan2(self, y, x):
        return np.atan2(y, x)
    
    def maximum(self, x, y):
        return np.maximum(x, y)
    
    def sum(self, x, *, axis, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    def clamp(self, x, min=None, max=None):
        return np.clip(x, a_min=min, a_max=max)
    
    def stack(self, seq, *, axis):
        return np.stack(seq, axis=axis)
    
    def broadcast_to(self, x, shape):
        return np.broadcast_to( x, shape)
    
    def transpose(self, x):
        return np.swapaxes(x, -1, -2)
