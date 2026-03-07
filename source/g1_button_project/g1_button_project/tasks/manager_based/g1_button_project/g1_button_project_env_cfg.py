import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg

from g1_button_project.tasks.manager_based.g1_button_project import mdp
from isaaclab.envs.mdp import JointEffortActionCfg
from g1_button_project.robots.g1_cfg import G1_CFG



@configclass
class G1ButtonProjectSceneCfg(InteractiveSceneCfg):
    """Configuration for G1 button project scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
    )

    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    button = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Button",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, -0.25, -0.07)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

@configclass
class ActionsCfg:
    """Действия: управление усилиями правой руки."""
    right_arm = JointEffortActionCfg(
        asset_name="robot", 
        joint_names=[
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", 
            "right_shoulder_yaw_joint", "right_elbow_joint", 
            "right_wrist_roll_joint", "right_wrist_pitch_joint", 
            "right_wrist_yaw_joint"
        ],
        scale=1.0 
    )

@configclass
class ObservationsCfg:
    """Наблюдения."""
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    reach_button = RewTerm(
        func=mdp.distance_to_button,
        weight=10.0,
        params={"button_name": "button", "ee_name": "right_wrist_yaw_link"},
    )

@configclass
class TerminationsCfg:
    """Терминация (заполним позже)."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class G1ButtonProjectEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self) -> None:
        # Общие настройки
        self.decimation = 2
        self.episode_length_s = 5.0
        
        # Конфиги
        self.scene = G1ButtonProjectSceneCfg(num_envs=2, env_spacing=4.0)
        self.observations = ObservationsCfg()
        self.actions = ActionsCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()

        # Настройки симуляции
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        events: EventCfg = EventCfg()

@configclass
class EventCfg:
    reset_button_position = EventTerm(
        func=mdp.reset_root_state_uniform, # <--- рандомизация
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("button"),
            "pose_range": {
                "x": (-0.15, 0.15), 
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),  
            },
            "velocity_range": {},
        },
    )