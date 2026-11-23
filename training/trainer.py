from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

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
    margin: float = 0.03
    temperature: float = 0.005
    fire_weight: float = 1.0
    run_dir: str = "runs"
    run_tag: str = ""
    eval_interval: int = 200
    eval_bsz: int = 64
    eval_seed: int = 1234


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

        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{self.cfg.run_tag}_{stamp}" if self.cfg.run_tag else stamp

        self.run_path = os.path.join(self.cfg.run_dir, name)
        self.ckpt_dir = self.cfg.checkpoint_dir

        self.eval_gen = torch.Generator(device=str(self.device))
        self.eval_gen.manual_seed(self.cfg.eval_seed)
        self.eval_batch = self._sample_eval(self.eval_gen)

        torch.manual_seed(self.cfg.seed)

    def _sample_eval(self, generator):
        bsz = self.cfg.eval_bsz
        device, dtype = self.device, torch.float32
        params = dict(
            self.sampler.sample(bsz, device, dtype), generator=generator, n_armors=4
        )
        t = self._sample_time(bsz, self.cfg.sql, device, dtype, generator=generator)
        yaw = torch.zeros(bsz, device=device, dtype=dtype)
        pitch = torch.zeros(bsz, device=device, dtype=dtype)
        return t, params, yaw, pitch

    def _init_log(self):
        os.makedirs(self.run_path, exist_ok=True)
        self.ckpt_dir = os.path.join(self.run_path, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(os.path.join(self.run_path, "config.json"), "w") as f:
            json.dump(
                {"train": asdict(self.cfg), "sampler": asdict(self.sampler.cfg)},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"run -> {self.run_path}")
        return SummaryWriter(self.run_path)

    def _sample_time(self, bsz, sql, device, dtype, generator=None):
        fps = torch.empty(bsz, 1, device=device, dtype=dtype).uniform_(
            *self.sampler.cfg.fps_range, generator=generator
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

        hit_loss = logs["hit_dist"].mean()

        fire_target = torch.sigmoid(
            (self.cfg.margin - logs["hit_dist"]) / self.cfg.temperature
        ).detach()
        fire_prob = logs["fire_prob"].clamp(1e-6, 1 - 1e-6)
        fire_loss = -(
            fire_target * torch.log(fire_prob)
            + (1 - fire_target) * torch.log(1 - fire_prob)
        )
        fire_loss = (fire_loss * mask).sum() / n_valid.clamp(min=1.0)

        loss = hit_loss + self.cfg.fire_weight * fire_loss
        return {
            "loss": loss,
            "hit_loss": hit_loss,
            "fire_loss": fire_loss,
        }

    @torch.no_grad()
    def _hit_stats(self, logs: dict):
        hit = logs["hit_dist"]
        update = logs["update"]
        stats = {
            "hit_dist": hit.mean().item(),
            "valid_ratio": update.float().mean().item(),
        }
        seen = hit[update]
        if seen.numel():
            q = torch.quantile(seen, torch.tensor([0.1, 0.5, 0.9], device=seen.device))
            stats["hit_p10"] = q[0].item()
            stats["hit_p50"] = q[1].item()
            stats["hit_p90"] = q[2].item()
            stats["hit_rate"] = (seen <= self.cfg.margin).float().mean().item()
        else:
            for key in ("hit_p10", "hit_p50", "hit_p90", "hit_rate"):
                stats[key] = float("nan")
        return stats

    def evaluate(self, writer: SummaryWriter, step: int) -> dict:
        self.sim.eval()
        t, params, yaw, pitch = self.eval_batch
        try:
            with torch.no_grad():
                logs = self.sim(t, params, yaw, pitch)
            stats = self._hit_stats(logs)
        finally:
            self.sim.train()
        for key, value in stats.items():
            writer.add_scalar(f"eval/{key}", value, step)
        return stats

    def train_step(self) -> dict:
        self.sim.train()
        self.optimizer.zero_grad()

        t, target_params, init_yaw, init_pitch = self._sample_batch()
        logs = self.sim(t, target_params, init_yaw, init_pitch)

        parts = self._compute_loss(logs)
        loss = parts["loss"]
        loss.backward()

        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params,
            self.cfg.grad_clip,
        )
        self.optimizer.step()

        m = self._hit_stats(logs)
        m["loss"] = loss.item()
        m["hit_loss"] = parts["hit_loss"].item()
        m["fire_loss"] = parts["fire_loss"].item()
        m["grad_norm"] = float(grad_norm)
        return m

    def _save(self, step):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"step_{step}.pt")
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
        writer = self._init_log()
        for step in range(1, self.cfg.num_steps + 1):
            m = self.train_step()

            if step % self.cfg.log_interval == 0:
                for key, value in m.items():
                    writer.add_scalar(f"train/{key}", value, step)
                writer.flush()
                print(
                    f"step {step:6d} | "
                    f"loss {m['loss']:.4f} | "
                    f"hit_dist {m['hit_dist']:.4f} | "
                    f"valid {m['valid_ratio']:.1%} | "
                    f"grad {m['grad_norm']:.3f}"
                )

            if step % self.cfg.eval_interval == 0:
                e = self.evaluate(writer, step)
                print(
                    f"eval  {step:6d} | "
                    f"hit50 {e['hit_p50']:.3f}m | "
                    f"hit {e['hit_rate']:.1%} | "
                    f"valid {e['valid_ratio']:.1%}"
                )

            if step % self.cfg.checkpoint_interval == 0:
                self._save(step)

        writer.close()
