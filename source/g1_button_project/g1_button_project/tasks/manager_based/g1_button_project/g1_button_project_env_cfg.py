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

import isaaclab.envs.mdp as i_mdp       # системные функции Isaac Lab - i_mdp
from . import mdp as p_mdp              # кастомные функции - p_mdp (Project MDP)

from isaaclab.envs.mdp import JointEffortActionCfg
from g1_button_project.robots.g1_cfg import G1_CFG


@configclass
class G1ButtonProjectSceneCfg(InteractiveSceneCfg):
    """Configuration for G1 button project scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
    )

    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    button = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Button",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True, # фиксация
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.275, -0.15, -0.07)),
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
        joint_pos = ObsTerm(func=i_mdp.joint_pos)
        joint_vel = ObsTerm(func=i_mdp.joint_vel)

        button_rel = ObsTerm(
            func=p_mdp.rel_button_pos,
            params={"button_name": "button", "ee_name": "right_hand_middle_ee"}
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    reach_button = RewTerm(
        func=p_mdp.distance_to_button,
        weight=50.0,
        params={"button_name": "button", "ee_name": "right_hand_middle_ee"},
    )

@configclass
class TerminationsCfg:
    """Терминация (заполним позже)."""
    time_out = DoneTerm(func=i_mdp.time_out, time_out=True)

@configclass
class EventCfg:
    # ГЛОБАЛЬНЫЙ СБРОС (сброс робота и кнопки к дефолтным init_state)
    reset_all = EventTerm(
        func=i_mdp.reset_scene_to_default,
        mode="reset",
        params={},
    )

    # РАНДОМИЗАЦИЯ КНОПКИ
    reset_button_position = EventTerm(
        func=i_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("button"),
            "pose_range": {
                "x": (-0.125, 0.125), # диапазон случайного разброса кнопки относительно init_state
                "y": (-0.25, 0.25),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )

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

        self.events = EventCfg()