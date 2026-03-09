import gymnasium as gym
from isaaclab.app import AppLauncher
import torch

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import g1_button_project.tasks  # Импортируем пакет задач, чтобы сработал gym.register

# Создание среды
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg("Template-G1-Button-Project-v0")
env = gym.make("Template-G1-Button-Project-v0", cfg=env_cfg)

print("Среда успешно создана!")

step_counter = 0

while simulation_app.is_running():
    with torch.inference_mode():
        if step_counter % 200 == 0:
            print(f"Сброс среды! Шаг: {step_counter}")
            env.reset()
        
        # Случайные действия
        actions = torch.rand(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
        
        step_counter += 1
        
        # Если нужно замедлить
        # import time; time.sleep(0.01)