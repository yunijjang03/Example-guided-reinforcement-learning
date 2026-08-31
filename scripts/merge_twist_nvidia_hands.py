# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge a TWIST body motion with NVIDIA G1 TriHand targets without modifying either source."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as functional


HAND_JOINT_NAMES = (
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--twist_motion",
        type=Path,
        default=Path("data_storage/TWIST/box/sub8_largebox_023.pkl"),
        help="TWIST motion containing the body trajectory.",
    )
    parser.add_argument(
        "--nvidia_episode",
        type=Path,
        default=Path("data_storage/g1_simple_high_var_lerobot/data/chunk-000/episode_000000.parquet"),
        help="NVIDIA LeRobot episode supplying the hand targets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_storage/TWIST/merged/sub8_largebox_023_nvidia_hands.pt"),
        help="New merged motion file.",
    )
    return parser.parse_args()


def main() -> None:
    """Create a merged body-and-hand trajectory while preserving source metadata."""
    args = parse_args()
    with args.twist_motion.open("rb") as file:
        twist = pickle.load(file)

    body_dof_pos = torch.as_tensor(np.asarray(twist["dof_pos"]), dtype=torch.float32)
    table = pq.read_table(args.nvidia_episode, columns=["action"])
    nvidia_actions = torch.as_tensor(np.asarray(table["action"].to_pylist()), dtype=torch.float32)
    if nvidia_actions.ndim != 2 or nvidia_actions.shape[1] != 32:
        raise RuntimeError(f"Expected NVIDIA actions with shape [frames, 32], got {tuple(nvidia_actions.shape)}.")

    hand_dof_pos = nvidia_actions[:, 14:28].transpose(0, 1).unsqueeze(0)
    hand_dof_pos = functional.interpolate(
        hand_dof_pos,
        size=body_dof_pos.shape[0],
        mode="linear",
        align_corners=True,
    ).squeeze(0).transpose(0, 1)

    merged = {
        "format_version": 1,
        "fps": float(twist["fps"]),
        "root_pos": torch.as_tensor(np.asarray(twist["root_pos"]), dtype=torch.float32),
        "root_rot_xyzw": torch.as_tensor(np.asarray(twist["root_rot"]), dtype=torch.float32),
        "body_dof_pos": body_dof_pos,
        "body_joint_count": body_dof_pos.shape[1],
        "hand_dof_pos": hand_dof_pos,
        "hand_joint_names": HAND_JOINT_NAMES,
        "source_twist_motion": str(args.twist_motion),
        "source_nvidia_episode": str(args.nvidia_episode),
        "nvidia_source_frames": nvidia_actions.shape[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(f"[INFO]: TWIST body frames: {body_dof_pos.shape[0]}")
    print(f"[INFO]: NVIDIA hand frames: {nvidia_actions.shape[0]} -> {hand_dof_pos.shape[0]}")
    print(f"[INFO]: Hand joints: {hand_dof_pos.shape[1]}")
    print(f"[INFO]: Saved merged motion to {args.output}")


if __name__ == "__main__":
    main()
