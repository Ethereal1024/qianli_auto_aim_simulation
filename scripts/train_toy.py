# scripts/train_toy.py
import torch
from torch import nn

from diff_env.simulator import AutoAimPolicy, DiffSimulator
from training.trainer import Trainer, TrainConfig


class ToyMLPPolicy(AutoAimPolicy):
    """最小可训练 policy，验证 trainer 流程。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8 + 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def init_state(self, bsz, device, dtype):
        return None

    def forward(
        self, keypoints, yaw_vel, pitch_vel, dt,
        cam_offset, muzzle_len, bullet_speed, state=None,
    ):
        x = torch.cat([
            keypoints,
            yaw_vel.unsqueeze(-1),
            pitch_vel.unsqueeze(-1),
            dt.unsqueeze(-1),
        ], dim=-1)
        raw = self.net(x)
        yaw_rate_cmd = torch.tanh(raw[..., 0])
        pitch_rate_cmd = torch.tanh(raw[..., 1])
        return yaw_rate_cmd, pitch_rate_cmd, None


if __name__ == "__main__":
    torch.manual_seed(0)
    policy = ToyMLPPolicy()
    sim = DiffSimulator(policy=policy)
    trainer = Trainer(sim, config=TrainConfig(num_steps=2000, log_interval=50))
    trainer.train()