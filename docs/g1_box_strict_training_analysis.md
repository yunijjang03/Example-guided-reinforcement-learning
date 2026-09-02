# Strict placement fine-tuning analysis

## Run

- Task: `Isaac-G1-Box-Place-Strict-Direct-v0`
- Starting checkpoint: `model_8496.pt`
- Iterations: 8,496 through 10,495 (2,000 updates)
- Parallel environments: 256
- Final checkpoint: `model_10495.pt`
- Wall-clock training time: 3,277.85 seconds

The strict task required the box to remain within 5 cm horizontally and 3 cm
vertically at less than 6 cm/s for 60 steps before release. Final validation
required 4.5 cm horizontal error, 2.5 cm vertical error, and less than 4 cm/s
after 75 release steps.

## Observed trend

The table contains averages over 100 training iterations. The first partial
bin is omitted because it contains only four initialization iterations.

| Iterations | Success | Success EMA | Goal distance [m] | Action std | Curriculum level |
|---|---:|---:|---:|---:|---:|
| 8500-8599 | 0.533 | 0.526 | 0.0465 | 1.448 | 1.143 |
| 8600-8699 | 0.495 | 0.500 | 0.0474 | 1.482 | 1.352 |
| 8700-8799 | 0.486 | 0.484 | 0.0473 | 1.514 | 1.308 |
| 8800-8899 | 0.499 | 0.500 | 0.0465 | 1.549 | 1.253 |
| 8900-8999 | 0.496 | 0.500 | 0.0460 | 1.583 | 1.290 |
| 9000-9099 | 0.497 | 0.494 | 0.0466 | 1.611 | 1.264 |
| 9100-9199 | 0.494 | 0.501 | 0.0466 | 1.640 | 1.297 |
| 9200-9299 | 0.499 | 0.495 | 0.0464 | 1.670 | 1.247 |
| 9300-9399 | 0.465 | 0.474 | 0.0496 | 1.699 | 1.169 |
| 9400-9499 | 0.474 | 0.470 | 0.0492 | 1.733 | 1.025 |
| 9500-9599 | 0.498 | 0.496 | 0.0468 | 1.765 | 1.157 |
| 9600-9699 | 0.489 | 0.499 | 0.0480 | 1.795 | 1.228 |
| 9700-9799 | 0.491 | 0.491 | 0.0474 | 1.826 | 1.301 |
| 9800-9899 | 0.493 | 0.498 | 0.0481 | 1.854 | 1.326 |
| 9900-9999 | 0.505 | 0.507 | 0.0467 | 1.881 | 1.332 |
| 10000-10099 | 0.495 | 0.492 | 0.0476 | 1.910 | 1.312 |
| 10100-10199 | 0.478 | 0.470 | 0.0480 | 1.941 | 1.197 |
| 10200-10299 | 0.479 | 0.472 | 0.0482 | 1.970 | 1.082 |
| 10300-10399 | 0.462 | 0.453 | 0.0441 | 1.998 | 0.921 |
| 10400-10495 | 0.226 | 0.227 | 0.0412 | 2.031 | 0.475 |

Across all 2,000 iterations, mean success was 0.477. The final iteration
reported 0.260 success and 0.185 success EMA. A single-iteration success rate
is noisy because it depends on which environments reset, so the 100-iteration
averages and EMA are more useful than the final sample alone.

## Why success fell

The strongest observed warning is monotonically increasing policy exploration.
Mean action standard deviation rose from 1.43 to 2.05. The final sharp success
drop coincided with the largest action standard deviation, making precise
settling and release harder. This is consistent with the non-zero PPO entropy
coefficient continuing to reward exploration during late fine-tuning. It is a
correlation from this run, not proof of a single cause.

The average goal distance improved from 4.67 cm in iterations 9900-9999 to
4.12 cm in the final bin while success dropped from 50.5% to 22.6%. Mean reward
also increased near the end. Therefore the policy was still rewarded for moving
and holding the box near the target, but it often failed the stricter temporal,
speed, or post-release validation. The dense shaping reward is not aligned
strongly enough with the binary final-placement objective.

Repeated failures reduced the average reverse-curriculum level from 1.33 to
0.48. This is a consequence of the degradation and creates a changing training
distribution, which makes raw training success less suitable for selecting the
best checkpoint.

## Checkpoint selection

Do not select `model_10495.pt` solely because it is the final checkpoint.
Evaluate at least `model_9900.pt`, `model_10000.pt`, `model_10200.pt`, and
`model_10495.pt` with deterministic actions, fixed initial-condition sets, and
the same strict success predicate. The training log suggests that checkpoints
around 9,900-10,000 are stronger candidates, but only a separate evaluation can
establish this.

## Recommended next run

1. Reduce late-stage exploration by lowering `entropy_coef` from `0.005` to
   `0.001` and use a lower fine-tuning learning rate such as `1e-4`.
2. Increase the terminal release-success reward and add an explicit penalty for
   completing release outside the final predicate. The current one-step success
   bonus is multiplied by the environment step time and is small relative to
   accumulated dense rewards.
3. Run deterministic evaluation at every saved checkpoint and use evaluation
   success for early stopping.
4. Report failures separately as position, height, speed, hold-duration, and
   post-release failures so the next regression has an identifiable cause.
5. Keep the strict task separate from the original task so the original
   placement checkpoint remains reproducible.

The aggregated source data is stored in
`docs/data/g1_box_strict_training_100_iteration.csv`.
