from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class SampleOffsetType(StrEnum):
    SAME = "same"
    BALANCE = "balance"
    VARIOUS = "various"


@dataclass
class SamplerConfig:
    distance_range: tuple[float, float] = (2.0, 8.0)
    lateral_range: tuple[float, float] = (-2.5, 2.5)
    height_range: tuple[float, float] = (-1.0, 1.0)

    yaw_rate_range: tuple[float, float] = (-6.0, 6.0)
    yaw_amp_range: tuple[float, float] = (0.0, 0.5)
    yaw_freq_range: tuple[float, float] = (0.3, 2.0)

    pitch_amp_range: tuple[float, float] = (0.0, 0.1)
    pitch_freq_range: tuple[float, float] = (0.3, 2.0)

    sinus_amp_range: tuple[float, float] = (0.0, 0.3)
    sinus_freq_range: tuple[float, float] = (0.2, 1.5)

    base_radius: float = 0.45
    radius_offset_type: SampleOffsetType = SampleOffsetType.BALANCE
    height_offset_type: SampleOffsetType = SampleOffsetType.BALANCE
    radius_offset_range: tuple[float, float] = (-0.15, 0.15)
    height_offset_range: tuple[float, float] = (-0.05, 0.05)

    fps_range: tuple[float, float] = (100.0, 300.0)


class TargetSampler:
    def __init__(
        self,
        config: SamplerConfig | None = None,
    ):
        self.cfg = config or SamplerConfig()

    def _uniform(self, bsz, range, device, dtype):
        lo, hi = range
        return torch.empty(bsz, device=device, dtype=dtype).uniform_(lo, hi)

    def sample(self, bsz: int, device, dtype) -> dict:
        cfg = self.cfg

        uniform_range = lambda x: self._uniform(bsz, x, device, dtype)

        p0 = torch.stack(
            [
                uniform_range(cfg.distance_range),
                uniform_range(cfg.lateral_range),
                uniform_range(cfg.height_range),
            ],
            dim=-1,
        )

        sinus_amp = torch.stack(
            [uniform_range(cfg.sinus_amp_range) for _ in range(3)], dim=-1
        )

        sinus_freq = torch.stack(
            [uniform_range(cfg.sinus_freq_range) for _ in range(3)], dim=-1
        )

        if cfg.radius_offset_type == SampleOffsetType.VARIOUS:
            radius_offset = torch.stack(
                [uniform_range(cfg.radius_offset_range) for _ in range(4)], dim=-1
            )
        elif cfg.radius_offset_type == SampleOffsetType.BALANCE:
            radius_sp1 = uniform_range(cfg.radius_offset_range)
            radius_sp2 = uniform_range(cfg.radius_offset_range)
            radius_offset = torch.stack(
                [radius_sp1, radius_sp2, radius_sp1, radius_sp2], dim=-1
            )
        else:
            radius_sp = uniform_range(cfg.radius_offset_range)
            radius_offset = torch.stack(
                [radius_sp, radius_sp, radius_sp, radius_sp], dim=-1
            )
            
        if cfg.height_offset_type == SampleOffsetType.VARIOUS:
                    height_offset = torch.stack(
                        [uniform_range(cfg.height_offset_range) for _ in range(4)], dim=-1
                    )
        elif cfg.height_offset_type == SampleOffsetType.BALANCE:
            height_sp1 = uniform_range(cfg.height_offset_range)
            height_sp2 = uniform_range(cfg.height_offset_range)
            height_offset = torch.stack(
                [height_sp1, height_sp2, height_sp1, height_sp2], dim=-1
            )
        else:
            height_sp = uniform_range(cfg.height_offset_range)
            height_offset = torch.stack(
                [height_sp, height_sp, height_sp, height_sp], dim=-1
            )

        return {
            "p0": p0,
            "yaw_rate": uniform_range(cfg.yaw_rate_range),
            "yaw_amp": uniform_range(cfg.yaw_amp_range),
            "yaw_freq": uniform_range(cfg.yaw_freq_range),
            "pitch_amp": uniform_range(cfg.pitch_amp_range),
            "pitch_freq": uniform_range(cfg.pitch_freq_range),
            "sinus_amp": sinus_amp,
            "sinus_freq": sinus_freq,
            "radius": cfg.base_radius,
            "radius_offset": radius_offset,
            "height_offset": height_offset,
        }
