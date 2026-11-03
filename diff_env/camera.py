import torch
from torch import nn

from tools.math import geometry as gmt


class Camera(nn.Module):
    def forward(
        self,
        points_gimbal: torch.Tensor,
        r_cg: torch.Tensor,
        t_cg: torch.Tensor,
        eps: float = 1e-6,
    ):        
        delta = points_gimbal - t_cg.unsqueeze(-2)
        points_cam = delta @ r_cg.mT
        
        valid = points_cam[..., 2] > eps
        
        keypoints = gmt.project_to_normalized(points_cam, eps)
        return keypoints, valid
