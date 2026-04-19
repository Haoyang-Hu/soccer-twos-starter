# HU PPO2 Shaped Reward Agent

**Agent name:** HU_PPO2_ShapedReward

## Description

PPO agent trained with `ppo_random_reward.py` against a random opponent with **dense reward shaping** added on top of the sparse goal signal. The shaping terms guide the agent to move the ball toward the opponent goal, maintain ball possession in the attacking half, and avoid idling — accelerating learning compared to the sparse-only baseline (HU_PPO1).

- **Training script:** `ppo_random_reward.py`
- **Opponent:** Random (uniform MultiDiscrete([3,3,3]) per player)
- **Reward:** Sparse goal signal + three dense shaping terms:
  - `+0.005 × Δball_x` — ball moves toward opponent goal (potential-based)
  - `+0.002 × ball_x/17` — ball in attacking half positional bonus
  - `-0.001/step` — stillness penalty when player speed < 0.1 m/s
- **Variation:** `team_vs_policy`, `multiagent=False`
- **Observation:** both players concatenated (336×2 = 672-dim)
- **Action:** MultiDiscrete([3,3,3,3,3,3]) — first 3 for player 0, next 3 for player 1

## Checkpoint

`ray_results/PPO_shaped/checkpoint-2500`

Trained for 20M timesteps (2500 iterations, batch size 8000). Final `episode_reward_mean` ≈ **1.61** (near the +2 single-goal ceiling).

## Batch script

`scripts/ppo_shaped_vs_random.batch`
