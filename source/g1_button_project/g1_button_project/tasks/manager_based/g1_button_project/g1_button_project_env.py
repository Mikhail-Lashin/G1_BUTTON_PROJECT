from isaaclab.envs import ManagerBasedRLEnv
from g1_button_project.tasks.g1_button_env_cfg import G1ButtonEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

class G1ButtonEnv(ManagerBasedRLEnv):
    cfg: G1ButtonEnvCfg

    def __init__(self, cfg: G1ButtonEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.robot = self.scene["robot"]

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)

        # маркер кнопки
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/button_marker",
            markers={
                "button": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.button_marker = VisualizationMarkers(marker_cfg)

        # вектор ошибки
        error_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/error_direction",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.2), # Настройте размер под руку робота
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # Зеленая стрелка
            ),
        },
    )
    self.error_marker = VisualizationMarkers(error_marker_cfg)

def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
    
    # 1. Получаем позиции
    button_pos, _ = self.scene["button"].get_world_poses()
    ee_idx = self.scene["robot"].find_bodies("right_hand_middle_ee")[0][0]
    ee_pos = self.scene["robot"].data.body_state_w[:, ee_idx, :3]
    
    # 2. Отрисовываем маркер кнопки (у вас уже есть)
    self.button_marker.visualize(button_pos)
    
    # 3. Отрисовываем стрелку из руки
    # Пока просто поместим её в руку, чтобы видеть, что она там. 
    # (Для разворота стрелки в сторону кнопки требуются кватернионы, 
    # оставим это на следующий шаг, если стрелка появится).
    self.error_marker.visualize(ee_pos)