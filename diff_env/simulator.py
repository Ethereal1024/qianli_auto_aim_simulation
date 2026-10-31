from typing import Any

import torch
from torch import nn


from diff_env.ballistic import Ballistic
from diff_env.camera import Camera
from diff_env.geometry import euler_to_rotmat
from diff_env.gimbal import Gimbal
from diff_env.target import TargetMotion

R_CG_DEFAULT = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class SimpleProportionalPolicy(nn.Module):
    def __init__(self, gain: float = 10.0):
        super().__init__()
        self.gain = gain

    def forward(
        self,
        keypoints: torch.Tensor,
        gimbal_rate: torch.Tensor,
        dt: torch.Tensor,
    ):
        cx = keypoints[..., 0::2].mean(-1, keepdim=True)
        cy = keypoints[..., 1::2].mean(-1, keepdim=True)

        yaw_rate_cmd = torch.tanh(-self.gain * cx)
        pitch_rate_cmd = torch.tanh(-self.gain * cy)
        return yaw_rate_cmd, pitch_rate_cmd


class DiffSimulator(nn.Module):
    def __init__(
        self,
        max_yaw_rate: float = 10.0,
        max_pitch_rate: float = 8.0,
        bullet_speed: float = 25.0,
        gravity: float = 9.81,
        muzzle_len: float = 0.15,
        armor_width: float = 0.12,
        armor_height: float = 0.06,
    ):
        super().__init__()
        self.gimbal = Gimbal(max_yaw_rate, max_pitch_rate)
        self.target = TargetMotion(armor_width, armor_height)
        self.camera = Camera()
        self.ballistic = Ballistic(bullet_speed, gravity)
        self.muzzle_len = muzzle_len
        self.r_cg: torch.Tensor
        self.register_buffer("r_cg", R_CG_DEFAULT.clone())

    def _world_to_gimbal(
        self,
        points_world: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ):
        zeros = torch.zeros_like(yaw)
        r_wg = euler_to_rotmat(yaw, -pitch, zeros)
        r_gw = r_wg.transpose(-1, -2)
        return torch.einsum("bij,bnj->bni", r_gw, points_world)

    def _gimbal_to_world(
        self,
        points_gimbal: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
    ):
        zeros = torch.zeros_like(yaw)
        r_wg = euler_to_rotmat(yaw, -pitch, zeros)
        return torch.einsum("bij,bnj->bni", r_wg, points_gimbal)

    def _mask_state(
        self, 
        old_state: torch.Tensor | list | tuple, 
        new_state: torch.Tensor | list | tuple, 
        mask: torch.Tensor
    ):
        if old_state is None:
            return new_state
        if isinstance(new_state, torch.Tensor) and isinstance(old_state, torch.Tensor):
            view_shape = (-1,) + (1,) * (new_state.dim() - 1)
            return torch.where(mask.view(*view_shape), new_state, old_state)
        if isinstance(new_state, (tuple, list)) and isinstance(old_state, (tuple, list)):
            return type(new_state)(
                self._mask_state(o, n, mask) for o, n in zip(old_state, new_state)
            )
        return new_state
    
    def _policy_step(
        self, 
        policy: nn.Module,
        kp: torch.Tensor, 
        gimbal_rate: float, 
        dt: float,
        state: torch.Tensor | list | tuple, 
        mask: torch.Tensor
    ):
        if not mask.any():
            zero = torch.zeros(kp.shape[0], 1, device=kp.device, dtype=kp.dtype)
            return zero, zero, state
        yr, pr, state_new = policy(kp, gimbal_rate, dt, state)
        yr = torch.where(mask.unsqueeze(-1), yr, torch.zeros_like(yr))
        pr = torch.where(mask.unsqueeze(-1), pr, torch.zeros_like(pr))
        state = self._mask_state(state, state_new, mask)
        return yr, pr, state

    def forward(
        self,
        t: torch.Tensor,
        target_params: dict,
        init_yaw: torch.Tensor,
        init_pitch: torch.Tensor,
        policy: nn.Module | None = None,
    ):
        if policy is None:
            policy = SimpleProportionalPolicy()

        bsz, sql = t.shape
        device = t.device
        dtype = t.dtype

        dt = torch.zeros_like(t)
        if sql > 1:
            dt[:, 1:] = t[:, 1:] - t[:, :-1]

        # target
        tgt = self.target(t, mode="complex", **target_params)
        corners_world = tgt["corners"]  # [bsz, sql, n_armors, 4, 3]
        centets_world = tgt["centers"]  # [bsz, sql, n_armors, 3]
        n_armors = corners_world.shape[2]

        # gimbal
        yaw = init_yaw.clone()
        pitch = init_pitch.clone()
        yaw_rate = torch.zeros(bsz, device=device, dtype=dtype)
        pitch_rate = torch.zeros(bsz, device=device, dtype=dtype)

        logs: dict = {
            "keypoints": [],
            "valid": [],
            "gimbal_yaw": [],
            "gimbal_pitch": [],
            "yaw_rate_cmd": [],
            "pitch_rate_cmd": [],
            "hit_dist": [],
            "t_fly": [],
        }

        r_cg = self.r_cg.unsqueeze(0).expand(bsz, -1, -1)

        for k in range(t):
            cnr_w = corners_world[:, k]
            ctr_w = centets_world[:, k]

            cg = self._world_to_gimbal(
                cnr_w.rashape(bsz, n_armors * 4, 3), yaw, pitch
            ).reshape(bsz, n_armors, 4, 3)
            crt_g = self._world_to_gimbal(ctr_w, yaw, pitch)

            kp, valid = self.camera(cg.reshape(bsz, n_armors * 4, 3), r_cg)
            kp = kp.reshape(bsz, n_armors, 4, 2)
            valid = valid.reshape(bsz, n_armors, 4)

            armor_valid = valid.all(dim=-1)

            dt_k = dt[:, k]
            yaw_cmds = torch.zeros(bsz, device=device, dtype=dtype)
            pitch_cmds = torch.zeros(bsz, device=device, dtype=dtype)

            for b in range(bsz):
                vis = armor_valid[b].nonzero(as_tuple=True)[0]
                if len(vis) == 0:
                    continue
                selected = vis[:2]
                # TODO: Currently we don't know if these two armors are literally visible ones.

                yr_list, pr_list = [], []
                for i, m in enumerate(selected):
                    kp_i = kp[b, m].rashape(1, 8)
                    if i == 0:
                        dt_i = dt_k[b].view(1, 1)
                    else:
                        dt_i = torch.zeros(1, 1, device=device, dtype=dtype)
                    gr_i = torch.stack([yaw_rate[b], pitch_rate[b]]).view(1, 2)
                    yr_i, pr_i = policy(kp_i, gr_i, dt_i)
                    yr_list.append(yr_i)
                    pr_list.append(pr_i)

                yaw_cmds[b] = torch.stack(yr_list).mean(0).squeeze(0)
                pitch_cmds[b] = torch.stack(pr_list).mean(0).squeeze(0)

            yaw, pitch, yaw_rate, pitch_rate = self.gimbal(
                yaw, pitch, yaw_cmds, pitch_cmds, dt_k
            )

            muzzle_local = torch.stack(
                [
                    torch.full_like(yaw, self.muzzle_len),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                ],
                dim=-1,
            )
            muzzle_world = self._gimbal_to_world(
                muzzle_local.unsqueeze(1), yaw, pitch
            ).squeeze(1)

            target_pos = ctr_w[:, 0]
            hit_pos, t_fly, hit_dist = self.ballistic(
                yaw, pitch, target_pos, muzzle_pos=muzzle_world
            )

            logs["keypoints"].append(kp)
            logs["valid"].append(armor_valid)
            logs["gimbal_yaw"].append(yaw)
            logs["gimbal_pitch"].append(pitch)
            logs["yaw_rate_cmd"].append(yaw_cmds)
            logs["pitch_rate_cmd"].append(pitch_cmds)
            logs["hit_dist"].append(hit_dist)
            logs["t_fly"].append(t_fly)

        for key, _ in logs.values():
            logs[key] = torch.stack(logs[key], dim=1)

        return logs
