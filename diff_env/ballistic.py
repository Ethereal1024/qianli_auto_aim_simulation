import torch
from torch import nn


class Ballistic(nn.Module):
    def __init__(self, bullet_speed: float = 25.0, gravity: float = 9.81):
        super().__init__()
        self.bullet_speed = bullet_speed
        self.gravity = gravity
        
    def forward(
        self,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        target_pos: torch.Tensor,
        muzzle_pos: torch.Tensor | None = None,
        iters: int = 5
    ):
        bsz = yaw.shape[0]
        device = yaw.device
        dtype = yaw.dtype
        
        if muzzle_pos is None:
            muzzle_pos = torch.zeros(bsz, 3, device=device, dtype=dtype)
            
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        direction = torch.stack(
            [cos_p * cos_y, cos_p * sin_y, sin_p], dim=-1
        )
        v0 = self.bullet_speed * direction
        g = torch.tensor([0.0, 0.0, -self.gravity], device=device, dtype=dtype)
        
        rel = target_pos - muzzle_pos
        d = torch.norm(rel, dim=-1)
        
        forward_proj = (v0 * rel).sum(-1)
        shootable = forward_proj > 0
        
        # initial estimate, ignoring gravity
        t = torch.where(
            shootable,
            forward_proj / (self.bullet_speed ** 2),
            torch.zeros_like(d)
        )
        t = torch.minimum(t, 2.0 * d / self.bullet_speed).clamp(min=0.0)
        
        # Newton iteration
        for _ in range(iters):
            pt = v0 * t.unsqueeze(-1) + 0.5 * g * (t.unsqueeze(-1) ** 2)
            u = pt - rel
            dp = v0 + g * t.unsqueeze(-1)
            
            f = (u * dp).sum(-1)  # dot multiply
            fp = (dp * dp).sum(-1) + (u * g).sum(-1)
            
            t = t - f / fp.clamp(min=1e-6)
            t = torch.minimum(t, 2.0 * d / self.bullet_speed).clamp(min=0.0)
            
        hit_pos = muzzle_pos + v0 * t.unsqueeze(-1) + 0.5 * g * (t.unsqueeze(-1) ** 2)
        hit_dist = torch.norm(hit_pos, dim=-1)
        
        return hit_pos, t, hit_dist