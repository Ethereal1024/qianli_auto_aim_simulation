import torch


def euler_to_rotmat(yaw, pitch, roll):
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    
    zero = torch.zeros_like(cy)
    one = torch.ones_like(cy)
    
    Rz = torch.stack([
        torch.stack([cy,    -sy,    zero], dim=-1),
        torch.stack([sy,    cy,     zero], dim=-1),
        torch.stack([zero,  zero,   one ], dim=-1),
    ], dim=-2)
    
    Ry = torch.stack([
        torch.stack([cp,    zero,   sp  ], dim=-1),
        torch.stack([zero,  one,    zero], dim=-1),
        torch.stack([-sp,    zero,  cp  ], dim=-1),
    ], dim=-2)
    
    Rx = torch.stack([
        torch.stack([one,   zero,   zero], dim=-1),
        torch.stack([zero,  cr,     -sr ], dim=-1),
        torch.stack([zero,  sr,     cr  ], dim=-1),
    ], dim=-2)
    
    return Rz @ Ry @ Rx

def rec_to_rotmat(rvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.norm(rvec, dim=-1, keepdim=True).clamp_min(eps)
    k = rvec / theta
    
    K = torch.zeros(*rvec.shape[:-1], 3, 3, device=rvec.device, dtype=rvec.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] =  k[..., 0]
    
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(K)
    
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R

def apply_transform(R: torch.Tensor, t: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    rotated = points @ R.mT
    return rotated + t.unsqueeze(-2)

def project_to_normalized(points_cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = points_cam[..., 2:3].clamp_min(eps)
    xn = points_cam[..., 0:1] / z
    yn = points_cam[..., 1:2] / z
    return torch.cat([xn, yn], dim=-1)

def compute_az_el(points_gimbal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = points_gimbal[..., 0]
    y = points_gimbal[..., 1]
    z = points_gimbal[..., 2]
    
    az = torch.atan(x, y)
    horizontal = torch.sqrt(x * x + y * y).clamp_min(eps)
    el = torch.atan2(z, horizontal)
    return torch.stack([az, el], dim=-1)
