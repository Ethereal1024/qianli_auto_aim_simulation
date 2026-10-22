import math

import torch
from torch import nn

from diff_env.geometry import euler_to_rotmat


class BodyMotion(nn.Module):
    def forward(self, t: torch.Tensor, params: dict):
        bsz, sln = t.shape
        device = t.device

        def _get(key: str, shape: tuple, default: float = 0.0):
            val = params.get(key, None)
            if val is None:
                return torch.full(shape, default, device=device, dtype=t.dtype)
            if isinstance(val, torch.Tensor):
                return val.to(device=device, dtype=t.dtype)
            return torch.full(shape, float(val), device=device, dtype=t.dtype)

        t_e = t.unsqueeze(-1)

        p0 = _get("p0", (bsz, 3))
        v = _get("v", (bsz, 3))
        a = _get("a", (bsz, 3))
        sinus_amp = _get("sinus_amp", (bsz, 3))
        sinus_freq = _get("sinus_freq", (bsz, 3))
        rw_sigma = _get("rw_sigma", (bsz, 3))

        pos = p0.unsqueeze(1) + v.unsqueeze(1) * t_e + 0.5 * a.unsqueeze(1) * t_e**2

        if sinus_amp.abs().sum() > 0:
            pos = pos + sinus_amp.unsqueeze(1) * torch.sin(
                2 * math.pi * sinus_freq.unsqueeze(1) * t_e
            )

        if rw_sigma.abs().sum() > 0:
            increments = torch.randn(
                bsz, sln, 3, device=device, dtype=t.dtype
            ) * rw_sigma.unsqueeze(1)
            pos = pos + torch.cumsum(increments, dim=1)

        yaw_0 = _get("yaw_0", (bsz,))
        yaw_rate = _get("yaw_rate", (bsz,))
        yaw_amp = _get("yaw_amp", (bsz,))
        yaw_freq = _get("yaw_freq", (bsz,))

        pitch_0 = _get("pitch_0", (bsz,))
        pitch_amp = _get("pitch_amp", (bsz,))
        pitch_freq = _get("pitch_freq", (bsz,))

        roll_0 = _get("roll_0", (bsz,))
        roll_amp = _get("roll_amp", (bsz,))
        roll_freq = _get("roll_freq", (bsz,))

        yaw = yaw_0.unsqueeze(1) + yaw_rate.unsqueeze(1) * t
        if yaw_amp.abs().sum() > 0:
            yaw = yaw + yaw_amp.unsqueeze(1) * torch.sin(
                2 * math.pi * yaw_freq.unsqueeze(1) * t
            )

        pitch = pitch_0.unsqueeze(1).expand(-1, sln)
        if pitch_amp.abs().sum() > 0:
            pitch = pitch + pitch_amp.unsqueeze(1) * torch.sin(
                2 * math.pi * pitch_freq.unsqueeze(1) * t
            )

        roll = roll_0.unsqueeze(1).expand(-1, sln)
        if roll_amp.abs().sum() > 0:
            roll = roll + roll_amp.unsqueeze(1) * torch.sin(
                2 * math.pi * roll_freq.unsqueeze(1) * t
            )

        rotations = euler_to_rotmat(yaw, pitch, roll)

        return {"positions": pos, "rotations": rotations}


class ArmorLayout(nn.Module):
    def __init__(self, armor_width: float = 0.12, armor_height: float = 0.06):
        super().__init__()
        self.armor_width = armor_width
        self.armor_height = armor_height

    def forward(
        self,
        n_armors: int,
        radius: torch.Tensor,
        radius_offsets: torch.Tensor | None = None,
        height_offsets: torch.Tensor | None = None,
        roll: float = math.pi * 5 / 12,
    ):
        device = radius.device
        bsz = radius.shape[0]

        if radius_offsets is None:
            radius_offsets = torch.zeros(
                bsz, n_armors, device=device, dtype=radius.dtype
            )
        if height_offsets is None:
            height_offsets = torch.zeros(
                bsz, n_armors, device=device, dtype=radius.dtype
            )

        theta = (
            torch.arange(n_armors, device=device, dtype=radius.dtype)
            * 2
            * math.pi
            / n_armors
        )

        r_eff = radius.unsqueeze(-1) + radius_offsets
        local_x = r_eff * torch.cos(theta)
        local_y = r_eff * torch.sin(theta)
        local_z = height_offsets
        local_positions = torch.stack([local_x, local_y, local_z], dim=-1)

        yaws = theta.unsqueeze(0).expand(bsz, -1)
        # vertical armor: width tangential, height up, normal radial outward
        zeros = torch.zeros_like(yaws)
        rolls = torch.full_like(yaws, roll)
        half_pi = torch.full_like(yaws, math.pi / 2)
        local_rotations = euler_to_rotmat(yaws + half_pi, zeros, rolls)

        w2 = self.armor_width / 2
        h2 = self.armor_height / 2
        corners_local = torch.tensor(
            [
                [-w2, -h2, 0.0],
                [w2, -h2, 0.0],
                [w2, h2, 0.0],
                [-w2, h2, 0.0],
            ],
            device=device,
            dtype=radius.dtype,
        )

        return local_positions, local_rotations, corners_local


class TargetMotion(nn.Module):
    def __init__(self, armor_width: float = 0.12, armor_height: float = 0.06):
        super().__init__()
        self.body_motion = BodyMotion()
        self.armor_layout = ArmorLayout(armor_width, armor_height)

    def forward(self, t, mode="static", **params):
        bsz, _ = t.shape
        device = t.device

        if mode in ("static", "constant_velocity"):
            n_armors = 1
        else:
            n_armors = params.get("n_armors", 4)

        radius = params.get("radius", None)
        if radius is None:
            radius = torch.full((bsz,), 0.2, device=device, dtype=t.dtype)
        elif not isinstance(radius, torch.Tensor):
            radius = torch.full((bsz,), float(radius), device=device, dtype=t.dtype)

        radius_offsets = params.get("armor_radius_offsets", None)
        height_offsets = params.get("armor_height_offsets", None)

        body = self.body_motion(t, params)
        body_pos = body["positions"]
        body_rot = body["rotations"]

        local_pos, local_rot, corners_local = self.armor_layout(
            n_armors, radius, radius_offsets, height_offsets
        )

        centers = body_pos.unsqueeze(2) + torch.einsum(
            "btij,bmj->btmi", body_rot, local_pos
        )
        world_rot = torch.einsum("btik,bmkj->btmij", body_rot, local_rot)
        corners = centers.unsqueeze(-2) + torch.einsum(
            "btmij,cj->btmci", world_rot, corners_local
        )
        # facing direction of each plate: local +Z is the plate normal
        normal_local = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=radius.dtype)
        world_normal = torch.einsum("btmij,j->btmi", world_rot, normal_local)
        armor_yaws = torch.atan2(world_normal[..., 1], world_normal[..., 0])

        return {
            "corners": corners,
            "centers": centers,
            "armor_yaws": armor_yaws,
            "armor_rotations": world_rot,
            "body_rotations": body_rot,
        }
