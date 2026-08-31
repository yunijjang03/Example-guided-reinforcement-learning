# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connect frozen G1 pickup and placement policies with a guarded transition."""

import argparse
import importlib.metadata as metadata
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pickup_checkpoint", type=Path, required=True, help="Pickup policy checkpoint.")
parser.add_argument("--placement_checkpoint", type=Path, required=True, help="Placement policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of evaluation environments.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum simulation steps; zero runs indefinitely.")
parser.add_argument("--switch_hold_steps", type=int, default=20, help="Stable grasp steps required before switching.")
parser.add_argument("--blend_steps", type=int, default=30, help="Action blending steps after switching.")
parser.add_argument("--lift_height", type=float, default=0.08, help="Required box lift height [m].")
parser.add_argument("--max_box_speed", type=float, default=0.40, help="Maximum switch box speed [m/s].")
parser.add_argument("--max_wrist_distance", type=float, default=0.40, help="Maximum wrist-to-box distance [m].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


TASK_NAME = "Isaac-G1-Box-Mimic-Direct-v0"


def _load_policy(env: RslRlVecEnvWrapper, agent_cfg, checkpoint: Path):
    """Load a frozen inference policy and preserve its observation normalizer."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    return runner.get_inference_policy(device=env.unwrapped.device)


def main() -> None:
    """Run pickup until a stable grasp, then blend into the placement policy."""
    env_cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.placement_only = False
    env_cfg.mixed_start = False

    gym_env = gym.make(TASK_NAME, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    pickup_policy = _load_policy(env, agent_cfg, args_cli.pickup_checkpoint)
    placement_policy = _load_policy(env, agent_cfg, args_cli.placement_checkpoint)

    task = env.unwrapped
    device = task.device
    switched = torch.zeros(task.num_envs, dtype=torch.bool, device=device)
    stable_steps = torch.zeros(task.num_envs, dtype=torch.long, device=device)
    blend_step = torch.zeros(task.num_envs, dtype=torch.long, device=device)
    transition_action = torch.zeros((task.num_envs, task.cfg.action_space), device=device)
    release_seen = torch.zeros(task.num_envs, dtype=torch.bool, device=device)
    switch_count = 0
    release_count = 0
    switched_failure_count = 0
    episode_count = 0

    obs = env.get_observations()
    step = 0
    print("[INFO]: Running hierarchical pickup-to-placement evaluation.")
    print("[INFO]: Waiting for a stable physical grasp before policy switching.")
    try:
        with torch.inference_mode():
            while simulation_app.is_running() and (args_cli.max_steps == 0 or step < args_cli.max_steps):
                pickup_action = pickup_policy(obs)

                new_release = task._released & switched & ~release_seen
                if torch.any(new_release):
                    release_seen |= new_release
                    release_count += int(new_release.sum().item())
                    print(f"[INFO]: Reached release in {int(new_release.sum().item())} environment(s) at step {step}.")

                box_pos = task._box.data.root_pos_w.torch
                box_speed = torch.linalg.vector_norm(task._box.data.root_lin_vel_w.torch, dim=-1)
                left_distance = torch.linalg.vector_norm(
                    task._robot.data.body_pos_w.torch[:, task._left_wrist_id] - box_pos, dim=-1
                )
                right_distance = torch.linalg.vector_norm(
                    task._robot.data.body_pos_w.torch[:, task._right_wrist_id] - box_pos, dim=-1
                )
                support_height = task.scene.env_origins[:, 2] + task.cfg.target_z
                lift_height = box_pos[:, 2] - support_height
                stable_grasp = (
                    (lift_height >= args_cli.lift_height)
                    & (box_speed <= args_cli.max_box_speed)
                    & (left_distance <= args_cli.max_wrist_distance)
                    & (right_distance <= args_cli.max_wrist_distance)
                    & ~switched
                )
                stable_steps = torch.where(stable_grasp, stable_steps + 1, torch.zeros_like(stable_steps))
                new_switch = (stable_steps >= args_cli.switch_hold_steps) & ~switched
                if torch.any(new_switch):
                    transition_action[new_switch] = pickup_action[new_switch]
                    switched |= new_switch
                    blend_step[new_switch] = 0
                    task._placement_episode[new_switch] = True
                    task._warmup_steps_remaining[new_switch] = 0
                    task.episode_length_buf[new_switch] = task.cfg.placement_warmup_steps
                    switch_count += int(new_switch.sum().item())
                    print(f"[INFO]: Switched {int(new_switch.sum().item())} environment(s) at step {step}.")
                    obs = env.get_observations()

                placement_action = placement_policy(obs)
                blend_step = torch.where(switched, blend_step + 1, blend_step)
                blend = (blend_step.float() / max(args_cli.blend_steps, 1)).clamp(0.0, 1.0).unsqueeze(-1)
                switched_action = torch.lerp(transition_action, placement_action, blend)
                actions = torch.where(switched.unsqueeze(-1), switched_action, pickup_action)
                obs, _, dones, _ = env.step(actions)

                done_mask = dones.bool()
                if torch.any(done_mask):
                    episode_count += int(done_mask.sum().item())
                    switched_failure_count += int((done_mask & switched & ~release_seen).sum().item())
                    switched[done_mask] = False
                    release_seen[done_mask] = False
                    stable_steps[done_mask] = 0
                    blend_step[done_mask] = 0
                    transition_action[done_mask] = 0.0
                    pickup_policy.reset(dones)
                    placement_policy.reset(dones)
                    if episode_count % 10 == 0:
                        print(
                            f"[INFO]: Episodes={episode_count}, switches={switch_count}, "
                            f"releases={release_count}, post-switch failures={switched_failure_count}."
                        )
                step += 1
    except KeyboardInterrupt:
        pass
    finally:
        print(
            f"[INFO]: Finished with episodes={episode_count}, switches={switch_count}, "
            f"releases={release_count}, post-switch failures={switched_failure_count}."
        )
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
