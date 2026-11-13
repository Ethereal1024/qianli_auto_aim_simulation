from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from diff_env.simulator import DiffSimulator

from .sampler import SamplerConfig, TargetSampler


@dataclass
class TrainConfig:
    num_steps: int = 10000
    bsz: int = 32
    sql: int = 50
    lr: float = 1e-3
    grad_clip: float = 1.0
    log_interval: int = 50
    checkpoint_interval: int = 2000
    checkpoint_dir: str = "checkpoints"
    seed: int = 0


class Trainer:
    def __init__(
        self,
        simulator: DiffSimulator,
        optimizer: torch.optim.Optimizer | None = None,
        config: TrainConfig | None = None,
        sampler_config: SamplerConfig | None = None,
        device: torch.device | None = None,
    ):
        self.sim = simulator
        self.cfg = config or TrainConfig()
        self.sampler = TargetSampler(sampler_config)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sim.to(self.device)

        trainable = [p for p in self.sim.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("Policy not trainable")
        self.optimizer = optimizer or torch.optim.Adam(trainable, lr=self.cfg.lr)
        torch.manual_seed(self.cfg.seed)

    def _sample_time(self, bsz, sql, device, dtype):
        fps = torch.empty(bsz, 1, device=device, dtype=dtype).uniform_(
            *self.sampler.cfg.fps_range
        )
        dt = 1.0 / fps
        t = torch.arange(sql, device=device, dtype=dtype).unsqueeze(0) * dt
        return t

    def _sample_batch(self):
        bsz, sql = self.cfg.bsz, self.cfg.sql
        device = self.device
        dtype = torch.float32

        params = self.sampler.sample(bsz, device, dtype)
        t = self._sample_time(bsz, sql, device, dtype)

        target_params = dict(params, n_armors=4)
        init_yaw = torch.zeros(bsz, device=device, dtype=dtype)
        init_pitch = torch.zeros(bsz, device=device, dtype=dtype)

        return t, target_params, init_yaw, init_pitch

    def _compute_loss(self, logs):
        mask = logs["update"].float()
        n_valid = mask.sum()
        hit_loss = (logs["hit_dist"] * mask).sum() / n_valid.clamp(min=1.0)
        return hit_loss, mask, n_valid

    def train_step(self) -> dict:
        self.sim.train()
        self.optimizer.zero_grad()

        t, target_params, init_yaw, init_pitch = self._sample_batch()
        logs = self.sim(t, target_params, init_yaw, init_pitch)

        loss, mask, n_valid = self._compute_loss(logs)
        loss.backward()

        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params,
            self.cfg.grad_clip,
        )
        self.optimizer.step()

        with torch.no_grad():
            hit_mean = ((logs["hit_dist"] * mask).sum() / n_valid.clamp(min=1.0)).item()
            valid_ratio = (n_valid / mask.numel()).item()

        return {
            "loss": loss.item(),
            "hit_dist": hit_mean,
            "valid_ratio": valid_ratio,
            "grad_norm": float(grad_norm),
        }

    def _save(self, step):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, f"step_{step}.pt")
        torch.save(
            {
                "step": step,
                "model": self.sim.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"saved -> {path}")

    def train(self):
        for step in range(1, self.cfg.num_steps + 1):
            m = self.train_step()

            if step % self.cfg.log_interval == 0:
                print(
                    f"step {step:6d} | "
                    f"loss {m['loss']:.4f} | "
                    f"hit_dist {m['hit_dist']:.4f} | "
                    f"valid {m['valid_ratio']:.1%} | "
                    f"grad {m['grad_norm']:.3f}"
                )

            if step % self.cfg.checkpoint_interval == 0:
                self._save(step)
