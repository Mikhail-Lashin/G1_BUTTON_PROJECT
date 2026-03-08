import torch
from isaaclab.envs import ManagerBasedRLEnv

def rel_button_pos(env: ManagerBasedRLEnv, button_name: str, ee_name: str) -> torch.Tensor:
    """Возвращает вектор от энд-эффектора до кнопки."""
    # 1. Позиция кнопки из буфера RigidObject [num_envs, 3]
    button_pos = env.scene[button_name].data.root_pos_w
    
    # 2. Позиция энд-эффектора из буфера Articulation [num_envs, 3]
    ee_idx = env.scene["robot"].find_bodies(ee_name)[0][0]
    ee_pos = env.scene["robot"].data.body_state_w[:, ee_idx, :3]
    
    # 3. Вектор ошибки
    return button_pos - ee_pos