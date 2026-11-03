import numpy as np
import torch

from .base import ArrayBackend
from .numpy_backend import NumpyBackend
from .torch_backend import TorchBackend

_NUMPY = NumpyBackend()
_TORCH = TorchBackend()

_REGISTRY: list[tuple[type, ArrayBackend]] = [
    (np.ndarray, _NUMPY),
    (torch.Tensor, _TORCH),
]


def array_namespace(*arrays) -> ArrayBackend:
    backends: set[ArrayBackend] = set()
    for a in arrays:
        for typ, backend in _REGISTRY:
            if isinstance(a, typ):
                backends.add(backend)
                break
        else:
            raise TypeError(f"Unsupported array type: {type(a).__name__}")
    if len(backends) > 1:
        names = {b.name for b in backends}
        raise TypeError(f"Mixed backends in one call: {names}")
    return backends.pop()
