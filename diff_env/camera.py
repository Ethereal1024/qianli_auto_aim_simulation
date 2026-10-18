import torch
from torch import nn

from diff_env.geometry import project_to_normalized, rvec_to_rotmat


class Camera(nn.Module):
    def forward(
        self,
        points_gimbal: torch.Tensor,
        rvec_cg: torch.Tensor,
        t_cg: torch.Tensor,
        eps: float = 1e-6,
    ):
        R_cg = rvec_to_rotmat(rvec_cg)
        
        delta = points_gimbal - t_cg.unsqueeze(-2)
        points_cam = delta @ R_cg.mT
        
        valid = points_cam[..., 2] > eps
        
        keypoints = project_to_normalized(points_cam, eps)
        return keypoints, valid
