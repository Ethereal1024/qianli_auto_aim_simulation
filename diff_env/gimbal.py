import torch
from torch import nn


class Gimbal(nn.Module):
    def __init__(self, max_yaw_rate: float = 10.0, max_pitch_rate: float = 8.0):
        super().__init__()
        self.max_yaw_rate = max_yaw_rate
        self.max_pitch_rate = max_pitch_rate

    def forward(
        self,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        yaw_rate_cmd: torch.Tensor,
        pitch_rate_cmd: torch.Tensor,
        dt: torch.Tensor,
    ):
        yaw_rate_actual = torch.clamp(
            yaw_rate_cmd, -self.max_yaw_rate, self.max_yaw_rate
        )
        pitch_rate_actual = torch.clamp(
            pitch_rate_cmd, -self.max_pitch_rate, self.max_pitch_rate
        )
        
        yaw_new = yaw + yaw_rate_actual * dt
        pitch_new = pitch + pitch_rate_actual * dt
        
        return yaw_new, pitch_new, yaw_rate_actual, pitch_rate_actual
