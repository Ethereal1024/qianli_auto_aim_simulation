import torch
from torch import nn


class Gimbal(nn.Module):
    def __init__(
        self, 
        max_yaw_rate: float = 10.0, 
        max_pitch_rate: float = 8.0, 
        tau: float = 0.03
    ):
        super().__init__()
        self.max_yaw_rate = max_yaw_rate
        self.max_pitch_rate = max_pitch_rate
        self.tau = tau

    def forward(
        self,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        yaw_vel_cmd: torch.Tensor,
        pitch_vel_cmd: torch.Tensor,
        yaw_rate_prev: torch.Tensor,
        pitch_rate_prev: torch.Tensor,
        dt: torch.Tensor,
    ):
        yaw_cmd_phys = torch.clamp(
            yaw_vel_cmd, -self.max_yaw_rate, self.max_yaw_rate
        )
        pitch_cmd_phys = torch.clamp(
            pitch_vel_cmd, -self.max_pitch_rate, self.max_pitch_rate
        )
        
        if self.tau > 0:
            alpha = 1.0 - torch.exp(-dt / self.tau)
            yaw_rate_actual = yaw_rate_prev + alpha * (yaw_cmd_phys - yaw_rate_prev)
            pitch_rate_actual = pitch_rate_prev + alpha * (pitch_cmd_phys - pitch_rate_prev)
        else:
            yaw_rate_actual = yaw_cmd_phys
            pitch_rate_actual = pitch_cmd_phys

        yaw_new = yaw + yaw_rate_actual * dt
        pitch_new = pitch + pitch_rate_actual * dt

        return yaw_new, pitch_new, yaw_rate_actual, pitch_rate_actual
