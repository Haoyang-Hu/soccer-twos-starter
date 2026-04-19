# HU PPO1 Baseline Agent

**Agent name:** HU_PPO1_Baseline

## Description

Baseline PPO agent trained with `example_ray_team_vs_random.py` against a random opponent using only the sparse goal reward (+2 / -2 / 0). No reward shaping. Serves as the baseline to compare against agents with improved reward design.

- **Training script:** `example_ray_team_vs_random.py`
- **Opponent:** Random (uniform MultiDiscrete([3,3,3]) per player)
- **Reward:** Sparse only — +2 on goal scored, -2 on goal conceded, 0 otherwise
- **Variation:** `team_vs_policy`, `multiagent=False`
- **Observation:** both players concatenated (336×2 = 672-dim)
- **Action:** MultiDiscrete([3,3,3,3,3,3]) — first 3 for player 0, next 3 for player 1

## Checkpoint

`ray_results/PPO_1/checkpoint_005000/checkpoint-5000`

Trained for 5000 iterations with batch size 12,000 and rollout fragment length 500.
