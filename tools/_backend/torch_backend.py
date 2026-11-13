from typing import Any

import torch

from .base import ArrayBackend


class TorchBackend(ArrayBackend):
    name = "torch"
    
    def zeros_like(self, x):
        return torch.zeros_like(x)
    
    def ones_like(self, x):
        return torch.ones_like(x)
    
    def zeros(self, shape, *, dlike):
        return torch.zeros(shape, dtype=dlike.dtype, device=dlike.device)
    
    def eye(self, n, *, dlike):
        return torch.eye(n, dtype=dlike.dtype, device=dlike.device)
    
    def cos(self, x):
        return torch.cos(x)
    
    def sin(self, x):
        return torch.sin(x)
    
    def sqrt(self, x):
        return torch.sqrt(x)
    
    def atan2(self, y, x):
        return torch.atan2(y, x)
    
    def maximum(self, x, y):
        return torch.maximum(x, y)
    
    def sum(self, x, *, axis, keepdims=False):
        return torch.sum(x, dim=axis, keepdim=keepdims)
    
    def norm(self, x, *, axis, keepdims=False):
        return torch.linalg.vector_norm(x, dim=axis, keepdim=keepdims)
    
    def clamp(self, x, min=None, max=None) -> Any:
        return torch.clamp(x, min=min, max=max)
    
    def stack(self, seq, *, axis):
        return torch.stack(seq, dim=axis)
    
    def broadcast_to(self, x, shape):
        return torch.broadcast_to(x, shape)
    
    def transpose(self, x) -> Any:
        return torch.transpose(x, -1, -2)