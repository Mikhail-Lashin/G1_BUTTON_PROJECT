from isaaclab.envs import ManagerBasedRLEnv
from g1_button_project.tasks.g1_button_env_cfg import G1ButtonEnvCfg
import isaaclab.sim as sim_utils

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

def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        
        real_button_pos, _ = self.scene["button"].get_world_poses()
        self.button_marker.visualize(real_button_pos)