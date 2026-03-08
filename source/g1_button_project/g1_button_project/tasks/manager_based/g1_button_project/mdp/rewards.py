# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def distance_to_button(env: ManagerBasedRLEnv, button_name: str, ee_name: str) -> torch.Tensor:
    # Теперь это RigidObject, берем данные из буфера напрямую
    button_pos = env.scene[button_name].data.root_pos_w
    
    ee_idx = env.scene["robot"].find_bodies(ee_name)[0][0]
    ee_pos = env.scene["robot"].data.body_state_w[:, ee_idx, :3]
    
    dist_sq = torch.sum(torch.square(button_pos - ee_pos), dim=-1)
    return -dist_sq