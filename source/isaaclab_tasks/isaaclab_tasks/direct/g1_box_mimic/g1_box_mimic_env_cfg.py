# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX


@configclass
class G1BoxMimicEnvCfg(DirectRLEnvCfg):
    """Configuration for fixed-base G1 box pick-and-place learning."""

    episode_length_s = 8.0
    decimation = 2
    action_space = 25
    observation_space = 120
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # The merged NVIDIA trajectory uses the Dex3/TriHand joint convention.
    robot: ArticulationCfg = G129_CFG_WITH_DEX3_BASE_FIX.replace(prim_path="/World/envs/env_.*/Robot")
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.22, 0.18),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.32, 0.12), roughness=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.79)),
    )
    pedestal: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.65, 0.70),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.23)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 0.35)),
    )

    motion_file = "data_storage/TWIST/merged/sub8_largebox_023_nvidia_hands.pt"
    action_scale = 0.35
    box_spawn_x_range = (-0.08, 0.08)
    box_spawn_y_range = (-0.12, 0.12)
    target_x_range = (0.33, 0.57)
    target_y_range = (-0.18, 0.18)
    target_z = 0.79
    target_radius_ranges = ((0.04, 0.07), (0.07, 0.12), (0.12, 0.18))
    target_curriculum_steps = (10000, 30000)
    box_half_height = 0.09
    goal_marker_half_height = 0.0075
    min_lift_height = 0.08
    success_distance = 0.10
    success_height_error = 0.06
    success_speed = 0.15
    success_hold_steps = 30
    final_success_distance = 0.10
    final_success_height_error = 0.06
    final_success_speed = 0.10
    release_duration_steps = 60
    goal_reward_distance_scale = 0.25
    box_max_distance = 1.5
    box_max_height = 2.0
    box_max_speed = 8.0
    max_debug_vis_envs = 4
    placement_only = False
    mixed_start = False
    placement_start_phase = 0.60
    placement_warmup_steps = 30
    placement_radius_ranges = ((0.00, 0.03), (0.02, 0.05), (0.04, 0.08), (0.07, 0.12), (0.10, 0.18))
    placement_start_probabilities = (0.40, 0.20, 0.0)
    mixed_curriculum_thresholds = ((0.10, 0.30), (0.25, 0.45))


@configclass
class G1BoxPlaceEnvCfg(G1BoxMimicEnvCfg):
    """Configuration for reverse-curriculum placement training."""

    episode_length_s = 6.0
    placement_only = True
    success_distance = 0.06
    success_height_error = 0.035
    success_speed = 0.08
    success_hold_steps = 45
    final_success_distance = 0.07
    final_success_height_error = 0.04
    final_success_speed = 0.06
    goal_reward_distance_scale = 0.15


@configclass
class G1BoxMixedEnvCfg(G1BoxPlaceEnvCfg):
    """Configuration for mixed pickup and strict placement training."""

    episode_length_s = 8.0
    placement_only = False
    mixed_start = True
