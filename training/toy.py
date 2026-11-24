import torch
from torch import nn

from diff_env.simulator import AutoAimPolicy


class ToyMLPPolicy(AutoAimPolicy):
    """最小可训练 policy，验证 trainer 流程。"""

    def __init__(self, n_armors: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_armors * 9 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def init_state(self, bsz, device, dtype):
        return None

    def forward(
        self, keypoints, yaw_vel, pitch_vel, dt,
        cam_offset, muzzle_len, bullet_speed, state=None, armor_mask=None
    ):
        if keypoints.dim() == 3:
            mask = armor_mask
            if mask is None:
                mask = torch.ones(
                    keypoints.shape[:2], dtype=torch.bool, device=keypoints.device
                )
            kp = (keypoints * mask.unsqueeze(-1)).flatten(1)
            extra = mask.to(keypoints.dtype)
        else:
            kp, extra = keypoints, None

        x = [kp, yaw_vel.unsqueeze(-1), pitch_vel.unsqueeze(-1), dt.unsqueeze(-1)]
        if extra is not None:
            x.insert(1, extra)
        raw = self.net(torch.cat(x, dim=-1))
        yaw_rate_cmd = torch.tanh(raw[..., 0])
        pitch_rate_cmd = torch.tanh(raw[..., 1])
        fire_logit = torch.sigmoid(raw[..., 2])
        return None, fire_logit, yaw_rate_cmd, pitch_rate_cmd