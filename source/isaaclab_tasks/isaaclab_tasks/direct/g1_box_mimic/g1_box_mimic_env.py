# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .g1_box_mimic_env_cfg import G1BoxMimicEnvCfg


BODY_REFERENCE_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
)
CONTROLLED_BODY_NAMES = BODY_REFERENCE_NAMES[12:]


class G1BoxMimicEnv(DirectRLEnv):
    """Train G1 to imitate a carry motion while physically moving a box to a commanded goal."""

    cfg: G1BoxMimicEnvCfg

    def __init__(self, cfg: G1BoxMimicEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        motion_path = Path(self.cfg.motion_file)
        if not motion_path.is_file():
            raise FileNotFoundError(f"Merged reference motion not found: {motion_path}")
        motion = torch.load(motion_path, map_location=self.device, weights_only=True)
        self._body_reference = motion["body_dof_pos"].to(self.device)
        self._hand_reference = motion["hand_dof_pos"].to(self.device)
        self._hand_names = tuple(motion["hand_joint_names"])
        self._reference = torch.cat((self._body_reference[:, 12:], self._hand_reference), dim=-1)

        self._controlled_names = CONTROLLED_BODY_NAMES + self._hand_names
        self._joint_ids = self._resolve_joint_ids(self._controlled_names)
        self._left_wrist_id = self._resolve_body_id("left_wrist_yaw_link")
        self._right_wrist_id = self._resolve_body_id("right_wrist_yaw_link")
        if len(self._joint_ids) != self.cfg.action_space or self._reference.shape[1] != self.cfg.action_space:
            raise RuntimeError("The configured action space does not match the merged motion joints.")

        limits = self._robot.data.soft_joint_pos_limits.torch[0, self._joint_ids]
        self._lower_limits = limits[:, 0]
        self._upper_limits = limits[:, 1]
        self._default_targets = self._robot.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._has_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._release_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._max_lift_height = torch.zeros(self.num_envs, device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._warmup_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_initialized = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._placement_episode = torch.full(
            (self.num_envs,), self.cfg.placement_only, dtype=torch.bool, device=self.device
        )
        self._normal_success_ema = 0.0
        self._placement_success_ema = 0.0
        self.set_debug_vis(self.num_envs <= self.cfg.max_debug_vis_envs)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._box = RigidObject(self.cfg.box)
        self.cfg.pedestal.spawn.func(
            self.cfg.pedestal.prim_path,
            self.cfg.pedestal.spawn,
            translation=self.cfg.pedestal.init_state.pos,
            orientation=self.cfg.pedestal.init_state.rot,
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["box"] = self._box
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_joint_ids(self, names: tuple[str, ...]) -> list[int]:
        ids = []
        for name in names:
            matches, _ = self._robot.find_joints(name)
            if len(matches) != 1:
                raise RuntimeError(f"Expected one G1 joint named {name}, got {matches}.")
            ids.append(matches[0])
        return ids

    def _resolve_body_id(self, name: str) -> int:
        matches, _ = self._robot.find_bodies(name)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one G1 body named {name}, got {matches}.")
        return matches[0]

    def _reference_at_phase(self, phase: torch.Tensor) -> torch.Tensor:
        position = phase * (self._reference.shape[0] - 1)
        index0 = position.long().clamp_max(self._reference.shape[0] - 1)
        index1 = (index0 + 1).clamp_max(self._reference.shape[0] - 1)
        blend = (position - index0).unsqueeze(-1)
        return torch.lerp(self._reference[index0], self._reference[index1], blend)

    def _phase(self) -> torch.Tensor:
        normal_phase = self.episode_length_buf.float() / max(self.max_episode_length - 1, 1)
        active_steps = (self.episode_length_buf - self.cfg.placement_warmup_steps).clamp_min(0).float()
        placement_episode_phase = active_steps / max(
            self.max_episode_length - self.cfg.placement_warmup_steps - 1, 1
        )
        placement_phase = self.cfg.placement_start_phase + (
            1.0 - self.cfg.placement_start_phase
        ) * placement_episode_phase
        return torch.where(self._placement_episode, placement_phase, normal_phase)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        reference = self._reference_at_phase(self._phase())
        targets = (reference + self.cfg.action_scale * self._actions).clamp(self._lower_limits, self._upper_limits)
        warming_up = self._warmup_steps_remaining > 0
        targets = torch.where(warming_up.unsqueeze(-1), reference, targets)
        if torch.any(self._released):
            release_ratio = (self._release_steps.float() / self.cfg.release_duration_steps).clamp(0.0, 1.0)
            retreat_targets = reference.clone()
            retreat_targets[:, 3:11] = self._reference[0, 3:11]
            retreat_targets[:, 11:] = self._default_targets[:, 11:]
            targets = torch.where(
                self._released.unsqueeze(-1),
                torch.lerp(targets, retreat_targets, release_ratio.unsqueeze(-1)),
                targets,
            )
        self._robot.set_joint_position_target_index(target=targets, joint_ids=self._joint_ids)
        if torch.any(warming_up):
            self._stabilize_box_in_grasp(warming_up)

    def _stabilize_box_in_grasp(self, env_mask: torch.Tensor) -> None:
        """Keep the dynamic box between the wrists while a placement episode settles."""
        wrist_midpoint = 0.5 * (
            self._robot.data.body_pos_w.torch[:, self._left_wrist_id]
            + self._robot.data.body_pos_w.torch[:, self._right_wrist_id]
        )
        carry_direction = wrist_midpoint - self._robot.data.root_pos_w.torch
        carry_direction[:, 2] = 0.0
        carry_direction /= torch.linalg.vector_norm(carry_direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        box_position = wrist_midpoint + 0.5 * self.cfg.box.spawn.size[0] * carry_direction
        box_yaw = torch.atan2(carry_direction[:, 1], carry_direction[:, 0])
        box_quat = torch.stack(
            (
                torch.zeros_like(box_yaw),
                torch.zeros_like(box_yaw),
                torch.sin(0.5 * box_yaw),
                torch.cos(0.5 * box_yaw),
            ),
            dim=-1,
        )
        box_pose = torch.cat((box_position, box_quat), dim=-1)
        env_ids = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        self._box.write_root_pose_to_sim_index(root_pose=box_pose[env_ids], env_ids=env_ids)
        self._box.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )

    def _get_observations(self) -> dict:
        phase = self._phase()
        root_pos = self._robot.data.root_pos_w.torch
        box_pos = self._box.data.root_pos_w.torch
        left_wrist = self._robot.data.body_pos_w.torch[:, self._left_wrist_id]
        right_wrist = self._robot.data.body_pos_w.torch[:, self._right_wrist_id]
        obs = torch.cat(
            (
                self._robot.data.joint_pos.torch[:, self._joint_ids],
                self._robot.data.joint_vel.torch[:, self._joint_ids] * 0.05,
                box_pos - root_pos,
                self._box.data.root_vel_w.torch,
                self._target_pos_w - box_pos,
                left_wrist - box_pos,
                right_wrist - box_pos,
                self._reference_at_phase(phase),
                torch.sin(2.0 * torch.pi * phase).unsqueeze(-1),
                torch.cos(2.0 * torch.pi * phase).unsqueeze(-1),
                self._actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        phase = self._phase()
        reference = self._reference_at_phase(phase)
        joint_error = torch.mean((self._robot.data.joint_pos.torch[:, self._joint_ids] - reference) ** 2, dim=-1)
        box_pos = self._box.data.root_pos_w.torch
        needs_goal = (~self._goal_initialized) & (self._warmup_steps_remaining <= 1)
        if torch.any(needs_goal):
            self._initialize_placement_goals(needs_goal, box_pos)
        box_speed = torch.linalg.vector_norm(self._box.data.root_lin_vel_w.torch, dim=-1)
        left_distance = torch.linalg.vector_norm(
            self._robot.data.body_pos_w.torch[:, self._left_wrist_id] - box_pos, dim=-1
        )
        right_distance = torch.linalg.vector_norm(
            self._robot.data.body_pos_w.torch[:, self._right_wrist_id] - box_pos, dim=-1
        )
        goal_offset = self._target_pos_w - box_pos
        goal_distance = torch.linalg.vector_norm(goal_offset, dim=-1)
        goal_xy_distance = torch.linalg.vector_norm(goal_offset[:, :2], dim=-1)
        goal_height_error = torch.abs(goal_offset[:, 2])
        goal_progress = (self._previous_goal_distance - goal_distance).clamp(-0.05, 0.05)
        self._previous_goal_distance.copy_(goal_distance)
        initial_height = self.scene.env_origins[:, 2] + self.cfg.target_z
        lift_height = (box_pos[:, 2] - initial_height).clamp(0.0, 0.35)
        self._max_lift_height = torch.maximum(self._max_lift_height, lift_height)
        self._has_lifted |= lift_height > self.cfg.min_lift_height
        action_rate = torch.mean((self._actions - self._previous_actions) ** 2, dim=-1)

        near_goal = (goal_xy_distance < self.cfg.success_distance) & (
            goal_height_error < self.cfg.success_height_error
        )
        settled = box_speed < self.cfg.success_speed
        active_success = near_goal & settled & self._has_lifted & ~self._released
        self._success_steps = torch.where(
            self._released, self._success_steps, torch.where(active_success, self._success_steps + 1, 0)
        )
        newly_succeeded = (self._success_steps >= self.cfg.success_hold_steps) & ~self._released
        self._released |= newly_succeeded
        self._release_steps = torch.where(self._released, self._release_steps + 1, self._release_steps)
        release_complete = self._release_steps >= self.cfg.release_duration_steps
        final_placement = (
            (goal_xy_distance < self.cfg.final_success_distance)
            & (goal_height_error < self.cfg.final_success_height_error)
            & (box_speed < self.cfg.final_success_speed)
        )
        self._episode_succeeded |= release_complete & final_placement
        warming_up = self._warmup_steps_remaining > 0
        self._warmup_steps_remaining = torch.where(
            warming_up, self._warmup_steps_remaining - 1, self._warmup_steps_remaining
        )
        carrying = self._has_lifted.float()
        imitation_scale = torch.where(self._has_lifted, 0.25, 1.25)
        goal_reward = 1.0 - torch.tanh(goal_distance / self.cfg.goal_reward_distance_scale)
        reward = (
            imitation_scale * torch.exp(-4.0 * joint_error)
            + (1.25 + carrying) * torch.exp(-12.0 * (left_distance + right_distance))
            + 3.0 * lift_height
            + carrying * (6.0 * goal_reward + 30.0 * goal_progress)
            + 5.0 * self._episode_succeeded.float()
            + 1.0 * (self._released & near_goal).float()
            - 0.02 * action_rate
            - carrying * 0.02 * box_speed.square().clamp_max(25.0)
        ) * self.step_dt
        return torch.where(warming_up, 0.0, reward)

    def _initialize_placement_goals(self, env_mask: torch.Tensor, box_pos: torch.Tensor) -> None:
        env_ids = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        levels = self._curriculum_level[env_ids]
        radius = torch.empty(len(env_ids), device=self.device)
        for level, radius_range in enumerate(self.cfg.placement_radius_ranges):
            level_mask = levels == level
            if torch.any(level_mask):
                radius[level_mask] = torch.empty(int(level_mask.sum().item()), device=self.device).uniform_(*radius_range)
        angle = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        target_xy = box_pos[env_ids, :2] + radius.unsqueeze(-1) * torch.stack(
            (torch.cos(angle), torch.sin(angle)), dim=-1
        )
        origins = self.scene.env_origins[env_ids]
        target_xy[:, 0].clamp_(self.cfg.target_x_range[0] + origins[:, 0], self.cfg.target_x_range[1] + origins[:, 0])
        target_xy[:, 1].clamp_(self.cfg.target_y_range[0] + origins[:, 1], self.cfg.target_y_range[1] + origins[:, 1])
        self._target_pos_w[env_ids, :2] = target_xy
        self._target_pos_w[env_ids, 2] = self.cfg.target_z + origins[:, 2]
        self._previous_goal_distance[env_ids] = torch.linalg.vector_norm(
            self._target_pos_w[env_ids] - box_pos[env_ids], dim=-1
        )
        self._goal_initialized[env_ids] = True
        if hasattr(self, "_goal_visualizer"):
            self._update_goal_visualizer()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        box_pos = self._box.data.root_pos_w.torch
        box_height = box_pos[:, 2] - self.scene.env_origins[:, 2]
        box_distance = torch.linalg.vector_norm(box_pos - self._robot.data.root_pos_w.torch, dim=-1)
        box_speed = torch.linalg.vector_norm(self._box.data.root_lin_vel_w.torch, dim=-1)
        invalid_state = ~torch.isfinite(self._box.data.root_state_w.torch).all(dim=-1)
        failed = (
            (box_height < 0.20)
            | (box_height > self.cfg.box_max_height)
            | (box_distance > self.cfg.box_max_distance)
            | (box_speed > self.cfg.box_max_speed)
            | invalid_state
        )
        release_complete = self._release_steps >= self.cfg.release_duration_steps
        return failed | release_complete, time_out

    def _placement_start_probability(self) -> float:
        """Return the placement-start probability for the mixed curriculum."""
        if not self.cfg.mixed_start:
            return float(self.cfg.placement_only)
        first_threshold, second_threshold = self.cfg.mixed_curriculum_thresholds
        if (
            self._normal_success_ema >= second_threshold[0]
            and self._placement_success_ema >= second_threshold[1]
        ):
            return self.cfg.placement_start_probabilities[2]
        if (
            self._normal_success_ema >= first_threshold[0]
            and self._placement_success_ema >= first_threshold[1]
        ):
            return self.cfg.placement_start_probabilities[1]
        return self.cfg.placement_start_probabilities[0]

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)

        final_goal_distance = torch.linalg.vector_norm(
            self._target_pos_w[env_ids] - self._box.data.root_pos_w.torch[env_ids], dim=-1
        )
        previous_placement = self._placement_episode[env_ids]
        previous_normal = ~previous_placement
        previous_success = self._episode_succeeded[env_ids]
        if torch.any(previous_normal):
            normal_success = previous_success[previous_normal].float().mean().item()
            self._normal_success_ema = 0.95 * self._normal_success_ema + 0.05 * normal_success
        if torch.any(previous_placement):
            placement_success = previous_success[previous_placement].float().mean().item()
            self._placement_success_ema = 0.95 * self._placement_success_ema + 0.05 * placement_success
        placement_probability = self._placement_start_probability()
        self.extras["log"] = {
            "Metrics/box_goal_distance": final_goal_distance.mean().item(),
            "Metrics/max_lift_height": self._max_lift_height[env_ids].mean().item(),
            "Metrics/success_rate": previous_success.float().mean().item(),
            "Metrics/curriculum_level": self._curriculum_level[env_ids].float().mean().item(),
            "Metrics/normal_success_ema": self._normal_success_ema,
            "Metrics/placement_success_ema": self._placement_success_ema,
            "Metrics/placement_start_probability": placement_probability,
        }
        if torch.any(previous_placement):
            placement_ids = env_ids[previous_placement]
            placement_success = previous_success[previous_placement]
            self._curriculum_level[placement_ids] = torch.where(
                placement_success,
                (self._curriculum_level[placement_ids] + 1).clamp_max(
                    len(self.cfg.placement_radius_ranges) - 1
                ),
                (self._curriculum_level[placement_ids] - 1).clamp_min(0),
            )
        self._robot.reset(env_ids)
        self._box.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._success_steps[env_ids] = 0
        self._has_lifted[env_ids] = False
        self._released[env_ids] = False
        self._release_steps[env_ids] = 0
        self._episode_succeeded[env_ids] = False
        self._max_lift_height[env_ids] = 0.0
        if self.cfg.placement_only:
            self._placement_episode[env_ids] = True
        elif self.cfg.mixed_start:
            self._placement_episode[env_ids] = torch.rand(len(env_ids), device=self.device) < placement_probability
        else:
            self._placement_episode[env_ids] = False
        placement_episode = self._placement_episode[env_ids]
        self._warmup_steps_remaining[env_ids] = torch.where(
            placement_episode,
            self.cfg.placement_warmup_steps,
            0,
        )
        self._goal_initialized[env_ids] = ~placement_episode

        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=self._robot.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        reset_phase = torch.where(
            placement_episode,
            self.cfg.placement_start_phase,
            0.0,
        )
        reset_reference = self._reference_at_phase(reset_phase)
        joint_pos[:, self._joint_ids] = reset_reference
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(
            velocity=self._robot.data.default_joint_vel.torch[env_ids], env_ids=env_ids
        )

        box_pose = self._box.data.default_root_pose.torch[env_ids].clone()
        box_pose[:, :3] += self.scene.env_origins[env_ids]
        box_pose[:, 0] += torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.box_spawn_x_range)
        box_pose[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.box_spawn_y_range)
        self._target_pos_w[env_ids] = box_pose[:, :3]
        self._previous_goal_distance[env_ids] = 0.0

        normal_episode = ~placement_episode
        if torch.any(normal_episode):
            normal_ids = env_ids[normal_episode]
            normal_box_pose = box_pose[normal_episode]
            box_xy_local = normal_box_pose[:, :2] - self.scene.env_origins[normal_ids, :2]
            if self.common_step_counter < self.cfg.target_curriculum_steps[0]:
                radius_range = self.cfg.target_radius_ranges[0]
            elif self.common_step_counter < self.cfg.target_curriculum_steps[1]:
                radius_range = self.cfg.target_radius_ranges[1]
            else:
                radius_range = self.cfg.target_radius_ranges[2]
            radius = torch.empty(len(normal_ids), device=self.device).uniform_(*radius_range)
            angle = torch.empty(len(normal_ids), device=self.device).uniform_(-torch.pi, torch.pi)
            target_xy = box_xy_local + radius.unsqueeze(-1) * torch.stack(
                (torch.cos(angle), torch.sin(angle)), dim=-1
            )
            target_xy[:, 0].clamp_(*self.cfg.target_x_range)
            target_xy[:, 1].clamp_(*self.cfg.target_y_range)
            self._target_pos_w[normal_ids, :2] = target_xy + self.scene.env_origins[normal_ids, :2]
            self._target_pos_w[normal_ids, 2] = self.cfg.target_z + self.scene.env_origins[normal_ids, 2]
            self._previous_goal_distance[normal_ids] = torch.linalg.vector_norm(
                self._target_pos_w[normal_ids] - normal_box_pose[:, :3], dim=-1
            )

        self._box.write_root_pose_to_sim_index(root_pose=box_pose, env_ids=env_ids)
        self._box.write_root_velocity_to_sim_index(
            root_velocity=self._box.data.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        if hasattr(self, "_goal_visualizer"):
            self._update_goal_visualizer()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/G1BoxMimic/goal"
                marker_cfg.markers["cuboid"].size = (0.20, 0.20, 0.015)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.1, 0.8, 0.2)
                self._goal_visualizer = VisualizationMarkers(marker_cfg)
            self._goal_visualizer.set_visibility(True)
        elif hasattr(self, "_goal_visualizer"):
            self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        self._update_goal_visualizer()

    def _update_goal_visualizer(self) -> None:
        marker_positions = self._target_pos_w.clone()
        marker_positions[:, 2] -= self.cfg.box_half_height - self.cfg.goal_marker_half_height
        self._goal_visualizer.visualize(marker_positions)
