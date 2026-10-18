import math

import torch
from torch import nn

from diff_env.geometry import euler_to_rotmat


class BodyMotion(nn.Module):
    def forward(self, t: torch.Tensor, params: dict):
        bsz, sln = t.shape
        device = t.device

        def _get(key: str, shape: tuple, default: float = 0.0):
            val = params.get(key, None)
            if val is None:
                return torch.full(shape, default, device=device, dtype=t.dtype)
            if isinstance(val, torch.Tensor):
                return val.to(device=device, dtype=t.dtype)
            return torch.full(shape, float(val), device=device, dtype=t.dtype)

        t_e = t.unsqueeze(-1)

        p0 = _get("p0", (bsz, 3))
        v = _get("v", (bsz, 3))
        a = _get("a", (bsz, 3))
        sinus_amp = _get("sinus_amp", (bsz, 3))
        sinus_freq = _get("sinus_freq", (bsz, 3))
        rw_sigma = _get("rw_sigma", (bsz, 3))
        
        pos = p0.unsqueeze(1) + v.unsqueeze(1) * t_e + 0.5 * a.unsqueeze(1) * t_e ** 2
        
        if sinus_amp.abs().sum() > 0:
            pos = pos + sinus_amp.unsqueeze(1) * torch.sin(
                2 * math.pi * sinus_freq.unsqueeze(1) * t_e
            )
