from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        resume: str | None = None,
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
        self.ckpt_dir = os.path.join(self.run_path, "checkpoints")

        self.eval_gen = torch.Generator(device=str(self.device))
        self.eval_gen.manual_seed(self.cfg.eval_seed)
        self.eval_batch = self._sample_eval(self.eval_gen)

        torch.manual_seed(self.cfg.seed)

        self.start_step = 0
        if resume:
            ckpt = torch.load(resume, map_location=self.device)
            self.sim.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            torch.set_rng_state(ckpt["rng"])
            if ckpt["cuda_rng"] is not None:
                torch.cuda.set_rng_state(ckpt["cuda_rng"])
            self.run_path = os.path.dirname(os.path.dirname(os.path.abspath(resume)))
            self.ckpt_dir = os.path.join(self.run_path, "checkpoints")
            self.start_step = int(ckpt["step"])
            print(f"resume <- {resume} at step {self.start_step}")

    def _sample_eval(self, generator):
        bsz = self.cfg.eval_bsz
        device, dtype = self.device, torch.float32
        params = dict(
            self.sampler.sample(bsz, device, dtype, generator=generator), n_armors=4
        )
        t = self._sample_time(bsz, self.cfg.sql, device, dtype, generator=generator)
        yaw = torch.zeros(bsz, device=device, dtype=dtype)
        pitch = torch.zeros(bsz, device=device, dtype=dtype)
        return t, params, yaw, pitch

    def _init_log(self):
        os.makedirs(self.run_path, exist_ok=True)
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
        fire_loss = F.binary_cross_entropy_with_logits(
            logs["fire_logit"], fire_target, reduction="none"
        )
        fire_loss = (fire_loss * mask).sum() / n_valid.clamp(min=1.0)

        loss = hit_loss + self.cfg.fire_weight * fire_loss
        return {
            "loss": loss,
            "hit_loss": hit_loss,
            "fire_loss": fire_loss,
        }

    @torch.no_grad()
    def _metrics(self, logs: dict) -> dict:
        update = logs["update"]
        seen = logs["hit_dist"][update]
        if not seen.numel():
            return {
                "hit_p50": float("nan"),
                "hit_rate": float("nan"),
                "fire_mean": float("nan"),
            }
        fire = logs["fire_prob"][update]
        hit_rate = (seen <= self.cfg.margin).float().mean().item()
        fire_mean = fire.mean().item()
        return {
            "hit_p50": torch.quantile(seen, 0.5).item(),
            "hit_rate": hit_rate,
            "fire_mean": fire_mean,
        }

    def _record(self, writer: SummaryWriter, prefix: str, stats: dict, step: int):
        for key, value in stats.items():
            writer.add_scalar(f"{prefix}/{key}", value, step)

    def evaluate(self) -> dict:
        self.sim.eval()
        t, params, yaw, pitch = self.eval_batch
        try:
            with torch.no_grad():
                logs = self.sim(t, params, yaw, pitch)
            return self._metrics(logs)
        finally:
            self.sim.train()

    def train_step(self) -> dict:
        self.sim.train()
        self.optimizer.zero_grad()

        t, target_params, init_yaw, init_pitch = self._sample_batch()
        logs = self.sim(t, target_params, init_yaw, init_pitch)

        parts = self._compute_loss(logs)
        parts["loss"].backward()

        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        self.optimizer.step()

        m = self._metrics(logs)
        m["loss"] = parts["loss"].item()
        m["hit_loss"] = parts["hit_loss"].item()
        m["fire_loss"] = parts["fire_loss"].item()
        m["grad_norm"] = float(grad_norm)
        return m

    def _save(self, step):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"step_{step}.pt")
        cuda_rng = (
            torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        )
        torch.save(
            {
                "step": step,
                "model": self.sim.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": json.dumps(
                    {"train": asdict(self.cfg), "sampler": asdict(self.sampler.cfg)}
                ),
                "rng": torch.get_rng_state(),
                "cuda_rng": cuda_rng,
            },
            path,
        )
        tqdm.write(f"saved -> {path}")

    def train(self):
        writer = self._init_log()
        last_time = time.perf_counter()
        last_step = self.start_step
        bar = tqdm(
            range(self.start_step + 1, self.cfg.num_steps + 1),
            total=self.cfg.num_steps,
            initial=self.start_step,
            desc=self.cfg.run_tag or "train",
            dynamic_ncols=True,
        )
        for step in bar:
            m = self.train_step()
            bar.set_postfix(
                {
                    "loss": f"{m['loss']:.4f}",
                    "hit50": f"{m['hit_p50']:.3f}m",
                    "hit": f"{m['hit_rate']:.1%}",
                    "fire": f"{m['fire_mean']:.3f}",
                    "grad": f"{m['grad_norm']:.3f}",
                },
                refresh=False,
            )

            if step % self.cfg.log_interval == 0:
                now = time.perf_counter()
                m["steps_per_sec"] = (step - last_step) / max(now - last_time, 1e-9)
                last_time, last_step = now, step
                self._record(writer, "train", m, step)
                writer.flush()
                tqdm.write(
                    f"step {step:6d} | loss {m['loss']:.4f} | "
                    f"hit50 {m['hit_p50']:.3f}m | hit {m['hit_rate']:.1%} | "
                    f"fire {m['fire_mean']:.3f} | "
                    f"grad {m['grad_norm']:.3f} | "
                    f"{m['steps_per_sec']:.0f} it/s"
                )

            if step % self.cfg.eval_interval == 0:
                e = self.evaluate()
                self._record(writer, "eval", e, step)
                tqdm.write(
                    f"eval  {step:6d} | hit50 {e['hit_p50']:.3f}m | "
                    f"hit {e['hit_rate']:.1%} | fire {e['fire_mean']:.3f}   "
                )
                last_time = time.perf_counter()
                last_step = step

            if step % self.cfg.checkpoint_interval == 0:
                self._save(step)

        writer.close()
