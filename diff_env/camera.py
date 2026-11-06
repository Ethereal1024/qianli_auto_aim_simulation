import torch
from torch import nn

from tools.math import geometry as gmt


class Camera(nn.Module):
    def forward(
        self,
        points_gimbal: torch.Tensor,
        r_gc: torch.Tensor,
        t_gc: torch.Tensor,
        eps: float = 1e-6,
    ):        
        points_cam = gmt.apply_transform_inverse(r_gc, t_gc, points_gimbal)
        valid = points_cam[..., 2] > eps
        
        keypoints = gmt.project_to_normalized(points_cam, eps)
        return keypoints, valid
