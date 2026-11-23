from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from tools.math import geometry as gmt
from dataclasses import dataclass

from .ballistic import Ballistic
from .camera import Camera
from .gimbal import Gimbal
from .target import TargetMotion

R_GC_DEFAULT = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class AutoAimPolicy(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init_state(self, bsz: int, device: torch.device, dtype: torch.dtype) -> Any: ...

    @abstractmethod
    def forward(
        self,
        keypoints: torch.Tensor,
        yaw_vel: torch.Tensor,
        pitch_vel: torch.Tensor,
        dt: torch.Tensor,
        cam_offset: torch.Tensor,
        muzzle_len: float,
        bullet_speed: float,
        state=None,
        armor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]: ...


class SimpleProportionalPolicy(AutoAimPolicy):
    def __init__(self, gain: float = 10.0):
        super().__init__()
        self.gain = gain

    def init_state(self, bsz: int, device: torch.device, dtype: torch.dtype):
        return None

    def forward(
        self,
        keypoints: torch.Tensor,
        yaw_vel: torch.Tensor,
        pitch_vel: torch.Tensor,
        dt: torch.Tensor,
        cam_offset: torch.Tensor,
        muzzle_len: float,
        bullet_speed: float,
        state=None,
        armor_mask: torch.Tensor | None = None,
    ):
        plate_cx = keypoints[..., 0::2].mean(-1, keepdim=True)
        plate_cy = keypoints[..., 1::2].mean(-1, keepdim=True)

        if plate_cx.dim() > 1:
            if armor_mask is None:
                armor_mask = torch.ones_like(plate_cx, dtype=torch.bool)
            w = armor_mask.to(plate_cx.dtype)
            n = w.sum(-1, keepdim=True).clamp(min=1.0)
            cx = (plate_cx * w).sum(-1, keepdim=True) / n
            cy = (plate_cy * w).sum(-1, keepdim=True) / n
            d2 = torch.where(
                armor_mask, plate_cx**2 + plate_cy**2, torch.full_like(plate_cx, 1e9)
            )
            d2_min = d2.min(-1, keepdim=True).values.clamp(max=1e8)
        else:
            cx, cy, d2_min = plate_cx, plate_cy, plate_cx**2 + plate_cy**2

        fire_prob = torch.sigmoid(-d2_min / 0.1 + 3).reshape(-1)

        yaw_vel_cmd = torch.tanh(-self.gain * cx).squeeze(-1)
        pitch_vel_cmd = torch.tanh(-self.gain * cy).squeeze(-1)
        return state, fire_prob, yaw_vel_cmd, pitch_vel_cmd


class DiffSimulator(nn.Module):
    def __init__(
        self,
        policy: AutoAimPolicy | None = None,
        cam_rotation: torch.Tensor | None = None,
        cam_offset: torch.Tensor | None = None,
        max_yaw_vel: float = 10.0,
        max_pitch_vel: float = 8.0,
        v_nominal: float = 5.0,
        bullet_speed: float = 25.0,
        gravity: float = 9.81,
        muzzle_len: float = 0.15,
        armor_width: float = 0.12,
        armor_height: float = 0.06,
        occlusion_factor: float = 0.0,
        keypoints_noise: float = 0.0,
    ):
        super().__init__()
        if policy is None:
            policy = SimpleProportionalPolicy()

        self.policy = policy
        self.gimbal = Gimbal(max_yaw_vel, max_pitch_vel)
        self.target = TargetMotion(armor_width, armor_height)
        self.camera = Camera()
        self.ballistic = Ballistic(bullet_speed, gravity)
        self.bullet_speed = bullet_speed
        self.muzzle_len = muzzle_len
        self.occlusion_factor = occlusion_factor
        self.armor_size = armor_height * armor_width
        self.keypoints_noise = keypoints_noise
        self.v_nominal = v_nominal

        r_gc_src = R_GC_DEFAULT if cam_rotation is None else cam_rotation
        self.r_gc: torch.Tensor
        self.register_buffer("r_gc", r_gc_src.clone())

        t_gc_src = torch.zeros(3) if cam_offset is None else cam_offset
        self.t_gc: torch.Tensor
        self.register_buffer("t_gc", t_gc_src.clone())

    def _muzzle_world(self, yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(yaw)
        muzzle_local = torch.stack(
            [torch.full_like(yaw, self.muzzle_len), zeros, zeros], dim=-1
        ).unsqueeze(
            1
        )  # [bsz, 1, 3]
        return self._gimbal_to_world(muzzle_local, yaw, pitch).squeeze(1)

    def _world_to_gimbal(
        self,
        points_world: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(yaw)
        r_wg = gmt.euler_to_rotmat(yaw, -pitch, zeros)
        r_gw = r_wg.transpose(-1, -2)
        return torch.einsum("bij,bnj->bni", r_gw, points_world)

    def _gimbal_to_world(
        self,
        points_gimbal: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(yaw)
        r_wg = gmt.euler_to_rotmat(yaw, -pitch, zeros)
        return torch.einsum("bij,bnj->bni", r_wg, points_gimbal)

    def _plate_centers(
        self, t_k: torch.Tensor, t_fly_all: torch.Tensor, target_params: dict
    ) -> torch.Tensor:
        future = self.target(
            t_k.unsqueeze(-1) + t_fly_all,  # [bsz, n_armors]
            mode="complex",
            **target_params,
        )  # [bsz, n_armors, n_armors, 3]
        return future["centers"].diagonal(dim1=1, dim2=2).movedim(-1, 1)

    def forward(
        self,
        t: torch.Tensor,
        target_params: dict,
        init_yaw: torch.Tensor,
        init_pitch: torch.Tensor,
    ):

        bsz, sql = t.shape
        device = t.device
        dtype = t.dtype

        dt = torch.zeros_like(t)
        if sql > 1:
            dt[:, 1:] = t[:, 1:] - t[:, :-1]

        # target
        tgt = self.target(t, mode="complex", **target_params)
        corners_world: torch.Tensor = tgt["corners"]  # [bsz, sql, n_armors, 4, 3]
        centers_world: torch.Tensor = tgt["centers"]  # [bsz, sql, n_armors, 3]
        n_armors = corners_world.shape[2]

        # gimbal
        yaw = init_yaw.clone()
        pitch = init_pitch.clone()
        yaw_vel = torch.zeros(bsz, device=device, dtype=dtype)
        pitch_vel = torch.zeros(bsz, device=device, dtype=dtype)

        state = self.policy.init_state(bsz, device, dtype)

        logs: dict = {
            "centers_world": centers_world,
            "corners_world": corners_world,
            "body_positions": tgt["body_positions"],
            "body_rotations": tgt["body_rotations"],
            "armor_rotations": tgt["armor_rotations"],
            "cam_rots": [],
            "cam_trans": [],
            "keypoints": [],
            "update": [],
            "gimbal_yaw": [],
            "gimbal_pitch": [],
            "yaw_vel": [],
            "pitch_vel": [],
            "fire_prob": [],
            "hit_pos": [],
            "target_pos": [],
            "hit_dist": [],
            "t_fly": [],
        }

        r_gc = self.r_gc.unsqueeze(0).expand(bsz, -1, -1)

        for k in range(sql):
            cnr_w = corners_world[:, k]  # [bsz, n_armors, 4, 3]
            ctr_w = centers_world[:, k]  # [bsz, n_armors, 3]

            cnr_g = self._world_to_gimbal(
                cnr_w.reshape(bsz, n_armors * 4, 3), yaw, pitch
            ).reshape(bsz, n_armors, 4, 3)
            # ctr_g = self._world_to_gimbal(ctr_w, yaw, pitch)

            keypoints, valid = self.camera(
                cnr_g.reshape(bsz, n_armors * 4, 3), r_gc, t_gc=self.t_gc
            )
            keypoints: torch.Tensor = keypoints.reshape(bsz, n_armors, 4, 2)
            valid: torch.Tensor = valid.reshape(bsz, n_armors, 4)
            keypoints = torch.where(
                valid.unsqueeze(-1), keypoints, torch.zeros_like(keypoints)
            )

            armor_visible: torch.Tensor = valid.all(dim=-1)  # [bsz, n_armors]

            r_wg: torch.Tensor = gmt.euler_to_rotmat(yaw, -pitch, torch.zeros_like(yaw))
            t_wc = r_wg @ self.t_gc

            normal = torch.cross(
                cnr_w[..., 1, :] - cnr_w[..., 0, :],
                cnr_w[..., 3, :] - cnr_w[..., 0, :],
                dim=-1,
            )
            diff = t_wc.unsqueeze(1) - ctr_w
            drct_dot_prod = (normal * diff).sum(-1)

            armor_front = drct_dot_prod > 0  # [bsz, n_armors]
            armor_visible = armor_visible & armor_front
            cost = 1 - (
                drct_dot_prod / (diff.norm(dim=-1).clamp(min=1e-6) * self.armor_size)
            ).clamp(0, 1)

            if self.training and self.occlusion_factor > 0:
                keep: torch.Tensor = (
                    torch.rand_like(armor_visible.float())
                    >= self.occlusion_factor * cost
                )
                armor_visible = armor_visible & keep

            keypoints = keypoints + torch.randn_like(keypoints) * self.keypoints_noise
            arange_batch = torch.arange(bsz, device=device)

            muzzle_prev = self._muzzle_world(yaw, pitch)  # [bsz, 3]
            dist_all = (ctr_w - muzzle_prev.unsqueeze(-2)).norm(
                dim=-1
            )  # [bsz, n_armors]
            dist_visible = torch.where(
                armor_visible, dist_all, torch.full_like(dist_all, float("inf"))
            )
            updated = armor_visible.any(dim=-1)
            idx = torch.where(
                updated, dist_visible.argmin(dim=-1), dist_all.argmin(dim=-1)
            )  # ? make sure we have at list one visible armor?

            kp_all = keypoints.reshape(bsz, n_armors, -1)  # [bsz, n_armors, 8]
            state_new, fire_prob, yv_cmd, pv_cmd = self.policy(
                keypoints=kp_all,
                armor_mask=armor_visible,
                yaw_vel=yaw_vel,
                pitch_vel=pitch_vel,
                dt=dt[:, k],
                cam_offset=self.t_gc,
                bullet_speed=self.bullet_speed,
                state=state,
                muzzle_len=self.muzzle_len,
            )
            state = state_new
            fire_prob = torch.where(updated, fire_prob, torch.zeros_like(fire_prob))

            yaw, pitch, yaw_vel, pitch_vel = self.gimbal(
                yaw,
                pitch,
                yv_cmd * self.v_nominal,
                pv_cmd * self.v_nominal,
                yaw_vel,
                pitch_vel,
                dt[:, k],
            )

            muzzle_world = self._muzzle_world(yaw, pitch)
            dist_now = (ctr_w[arange_batch, idx] - muzzle_world).norm(dim=-1)

            t_fly = dist_now.unsqueeze(-1).expand(bsz, n_armors) / self.bullet_speed
            yaw_n = yaw.unsqueeze(-1).expand(bsz, n_armors).reshape(-1)
            pitch_n = pitch.unsqueeze(-1).expand(bsz, n_armors).reshape(-1)
            muzzle_n = muzzle_world.unsqueeze(1).expand(bsz, n_armors, 3).reshape(-1, 3)

            for _ in range(3):
                centers_n = self._plate_centers(t[:, k], t_fly, target_params)
                hit_pos_n, t_n, _ = self.ballistic(
                    yaw_n,
                    pitch_n,
                    centers_n.reshape(-1, 3),
                    muzzle_pos=muzzle_n,
                    iters=3,
                )
                t_fly = t_n.reshape(bsz, n_armors)

            centers_final = self._plate_centers(t[:, k], t_fly, target_params)
            hit_pos_n = hit_pos_n.reshape(bsz, n_armors, 3)  # type: ignore
            dist_plates = torch.norm(hit_pos_n - centers_final, dim=-1)

            hit_dist, best = dist_plates.min(dim=-1)
            hit_pos = hit_pos_n[arange_batch, best]
            target_pos_final = centers_final[arange_batch, best]
            t_fly = t_fly[arange_batch, best]

            logs["cam_rots"].append(r_wg @ self.r_gc)
            logs["cam_trans"].append(t_wc)
            logs["keypoints"].append(keypoints)
            logs["update"].append(updated)
            logs["gimbal_yaw"].append(yaw)
            logs["gimbal_pitch"].append(pitch)
            logs["yaw_vel"].append(yaw_vel)
            logs["pitch_vel"].append(pitch_vel)
            logs["fire_prob"].append(fire_prob)
            logs["hit_pos"].append(hit_pos)
            logs["target_pos"].append(target_pos_final)
            logs["hit_dist"].append(hit_dist)
            logs["t_fly"].append(t_fly)

        for key in list(logs.keys()):
            if isinstance(logs[key], list):
                logs[key] = torch.stack(logs[key], dim=1)

        return logs
