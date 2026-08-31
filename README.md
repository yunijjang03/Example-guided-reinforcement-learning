# Example-Guided Reinforcement Learning for G1 Box Manipulation

Isaac Lab experiments for physics-based Unitree G1 box manipulation guided by a reference motion.

The project combines:

- motion-imitation rewards inspired by DeepMimic,
- PPO training with RSL-RL,
- randomized box pickup,
- reverse-curriculum placement training,
- strict post-release success checks, and
- hierarchical switching between frozen pickup and placement policies.

## Current policies

The checkpoints are intentionally not committed to Git.

- Pickup policy: `model_4498.pt`
- Strict placement policy: `model_8496.pt`

The hierarchical evaluator runs the pickup policy until a stable physical grasp is detected, then blends into the placement policy.

## Repository layout

- `source/isaaclab_tasks/.../g1_box_mimic/`: Isaac Lab task and PPO configuration
- `scripts/play_g1_box_hierarchical.py`: pickup-to-placement policy switching
- `scripts/merge_twist_nvidia_hands.py`: reference-motion preprocessing

Detailed installation, data preparation, training, and streaming commands are included on the feature branch with the implementation.

## Data and checkpoints

TWIST/NVIDIA source motion data, generated motion tensors, training logs, and model checkpoints are excluded because of size and licensing considerations.

## License

BSD-3-Clause. See `LICENSE`.
