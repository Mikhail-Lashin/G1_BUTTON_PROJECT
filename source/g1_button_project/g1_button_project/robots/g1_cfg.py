import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ml/g1_button_project/g1_button_project/source/g1_button_project/g1_button_project/robots/g1.usd", 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint"
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "lock_body": ImplicitActuatorCfg(
            joint_names_expr=["(?!right_(shoulder|elbow|wrist)).*"], 
            effort_limit_sim=200.0,
            velocity_limit_sim=100.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)