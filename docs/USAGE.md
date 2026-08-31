# Usage

## Requirements

- NVIDIA Isaac Lab with Isaac Sim
- RSL-RL support enabled
- Unitree G1 Dex3/TriHand asset configuration
- CUDA-capable NVIDIA GPU

The code follows the Isaac Lab source-tree layout. Copy the contents of this repository into the root of a compatible Isaac Lab checkout.

## Reference motion

The environment expects the merged reference motion at:

```text
data_storage/TWIST/merged/sub8_largebox_023_nvidia_hands.pt
```

Generate it with:

```bash
./isaaclab.sh -p scripts/merge_twist_nvidia_hands.py --help
```

The source TWIST and NVIDIA motion datasets are not distributed by this repository. Obtain and use them according to their respective licenses.

## Registered tasks

- `Isaac-G1-Box-Mimic-Direct-v0`: randomized pickup, transport, placement, and release
- `Isaac-G1-Box-Place-Direct-v0`: placement-only reverse curriculum
- `Isaac-G1-Box-Mixed-Direct-v0`: experimental mixed-start curriculum

## Placement training

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-G1-Box-Place-Direct-v0 \
  --num_envs 128 \
  --max_iterations 2000 \
  --resume \
  --load_run <pickup-run> \
  --checkpoint model_4498.pt \
  --viz none
```

The strict placement success check requires the box to remain near the target with low velocity after the scripted hand release.

## Hierarchical pickup-to-placement evaluation

The evaluator keeps both policies frozen and preserves each checkpoint's observation normalizer.

```bash
PUBLIC_IP=<server-ip> ./isaaclab.sh -p \
  scripts/play_g1_box_hierarchical.py \
  --pickup_checkpoint logs/rsl_rl/g1_box_mimic/<pickup-run>/model_4498.pt \
  --placement_checkpoint logs/rsl_rl/g1_box_mimic/<placement-run>/model_8496.pt \
  --num_envs 1 \
  --livestream 1 \
  --viz kit
```

Default policy-switch conditions:

- box lifted at least 0.08 m,
- box linear speed at most 0.40 m/s,
- each wrist within 0.40 m of the box,
- all conditions held for 20 simulation steps, and
- actions blended for 30 steps after switching.

## Reproduced experiment result

In a 16-environment, 1,200-step connection test:

- 29 stable-grasp policy switches were detected,
- 10 switched trajectories reached the release stage, and
- 15 switched trajectories terminated before release.

This demonstrates that the learned skills can connect, but the transition distribution still needs improvement before distillation or deployment.

## Excluded artifacts

The following are intentionally not committed:

- model checkpoints,
- TensorBoard logs,
- rendered videos,
- generated motion tensors, and
- original motion datasets.
