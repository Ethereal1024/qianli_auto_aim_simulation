from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from tools.math import geometry as gmt

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
    ) -> Any: ...


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
    ):
        cx = keypoints[..., 0::2].mean(-1, keepdim=True)
        cy = keypoints[..., 1::2].mean(-1, keepdim=True)

        yaw_vel_cmd = torch.tanh(-self.gain * cx).squeeze(-1)
        pitch_vel_cmd = torch.tanh(-self.gain * cy).squeeze(-1)
        return yaw_vel_cmd, pitch_vel_cmd, None


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

    def _partial_update_tensor(
        self,
        old: torch.Tensor,
        new: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        view_shape = (-1,) + (1,) * (new.dim() - 1)
        return torch.where(mask.view(*view_shape), new, old)

    def _partial_update(
        self,
        old: torch.Tensor | list | tuple,
        new: torch.Tensor | list | tuple,
        mask: torch.Tensor,
    ):
        if old is None:
            return new
        if isinstance(new, torch.Tensor) and isinstance(old, torch.Tensor):
            return self._partial_update_tensor(old, new, mask)
        if isinstance(new, (tuple, list)) and isinstance(old, (tuple, list)):
            return type(new)(self._partial_update(o, n, mask) for o, n in zip(old, new))
        return new

    def _world_to_gimbal(
        self,
        points_world: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ):
        zeros = torch.zeros_like(yaw)
        r_wg = gmt.euler_to_rotmat(yaw, -pitch, zeros)
        r_gw = r_wg.transpose(-1, -2)
        return torch.einsum("bij,bnj->bni", r_gw, points_world)

    def _gimbal_to_world(
        self,
        points_gimbal: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ):
        zeros = torch.zeros_like(yaw)
        r_wg = gmt.euler_to_rotmat(yaw, -pitch, zeros)
        return torch.einsum("bij,bnj->bni", r_wg, points_gimbal)

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

        yv_cmd = torch.zeros(bsz, dtype=torch.bool, device=device)
        pv_cmd = torch.zeros(bsz, dtype=torch.bool, device=device)

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

            dt_used = torch.zeros(bsz, dtype=torch.bool, device=device)
            keypoints = keypoints + torch.randn_like(keypoints) * self.keypoints_noise

            for m in range(n_armors):
                batch_update = armor_visible[..., m]  # [bsz,]
                dt_for_policy = torch.where(
                    dt_used, torch.zeros_like(dt[:, k]), dt[:, k]
                )

                yv_cmd_new, pv_cmd_new, state_new = self.policy(
                    keypoints=keypoints[:, m].reshape(bsz, 8),
                    yaw_vel=yaw_vel,
                    pitch_vel=pitch_vel,
                    dt=dt_for_policy,
                    cam_offset=self.t_gc,
                    bullet_speed=self.bullet_speed,
                    state=state,
                    muzzle_len=self.muzzle_len,
                )

                yv_cmd = self._partial_update_tensor(yv_cmd, yv_cmd_new, batch_update)
                pv_cmd = self._partial_update_tensor(pv_cmd, pv_cmd_new, batch_update)
                state = (
                    self._partial_update(state, state_new, batch_update)
                    if state is not None
                    else None
                )

                dt_used = dt_used | batch_update

            # 云台积分
            yaw_new, pitch_new, yaw_vel_new, pitch_vel_new = self.gimbal(
                yaw,
                pitch,
                yv_cmd * self.v_nominal,
                pv_cmd * self.v_nominal,
                yaw_vel,
                pitch_vel,
                dt[:, k],
            )

            updated = armor_visible.any(dim=-1)
            yaw = self._partial_update_tensor(yaw, yaw_new, updated)
            pitch = self._partial_update_tensor(pitch, pitch_new, updated)
            yaw_vel = self._partial_update_tensor(yaw_vel, yaw_vel_new, updated)
            pitch_vel = self._partial_update_tensor(pitch_vel, pitch_vel_new, updated)

            # 枪口世界位置
            muzzle_local = torch.stack(
                [
                    torch.full_like(yaw, self.muzzle_len),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                ],
                dim=-1,
            ).unsqueeze(
                1
            )  # [B, 1, 3]
            muzzle_world = self._gimbal_to_world(muzzle_local, yaw, pitch).squeeze(
                1
            )  # [B, 3]

            rel_all = ctr_w - muzzle_world.unsqueeze(-2)  # [bsz, n_armors, 3]
            scores = rel_all.norm(dim=-1)  # [bsz, n_armors]
            scores = torch.where(
                armor_visible, scores, torch.full_like(scores, float("inf"))
            )

            idx = scores.argmin(dim=-1)  # [bsz,]
            dist_now = scores.gather(1, idx.unsqueeze(-1)).squeeze(-1)  # [bsz,]
            dist_now = torch.where(updated, dist_now, torch.zeros_like(dist_now))
            t_fly = dist_now / self.bullet_speed

            arange_batch = torch.arange(bsz, device=device)

            hit_pos, hit_dist = None, None
            for _ in range(3):
                future = self.target(
                    (t[:, k] + t_fly).unsqueeze(-1),
                    mode="complex",
                    **target_params,
                )
                future_ctr = future["centers"][:, 0]
                target_pos = future_ctr[arange_batch, idx]

                hit_pos, t_fly, hit_dist = self.ballistic(
                    yaw, pitch, target_pos, muzzle_pos=muzzle_world, iters=3
                )

            future_final = self.target(
                (t[:, k] + t_fly).unsqueeze(-1),
                mode="complex",
                **target_params,
            )
            target_pos_final = future_final["centers"][:, 0][arange_batch, idx]
            hit_dist = torch.norm(hit_pos - target_pos_final, dim=-1)

            logs["cam_rots"].append(r_wg @ self.r_gc)
            logs["cam_trans"].append(t_wc)
            logs["keypoints"].append(keypoints)
            logs["update"].append(updated)
            logs["gimbal_yaw"].append(yaw)
            logs["gimbal_pitch"].append(pitch)
            logs["yaw_vel"].append(yaw_vel)
            logs["pitch_vel"].append(pitch_vel)
            logs["hit_pos"].append(hit_pos)
            logs["target_pos"].append(target_pos_final)
            logs["hit_dist"].append(hit_dist)
            logs["t_fly"].append(t_fly)

        for key in list(logs.keys()):
            if isinstance(logs[key], list):
                logs[key] = torch.stack(logs[key], dim=1)

        return logs
