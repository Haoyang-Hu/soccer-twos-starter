# HU PPO3 Self-Play Agent

**Agent name:** PPO3SelfPlayAgent

## Description

PPO agent trained with `ppo_selfplay.py` using **frozen self-play** as the opponent and an enhanced 7-term dense reward shaping scheme. The agent is warm-started from the shaped-reward random-opponent checkpoint (HU_PPO2, ~20 M steps) and then trained against a frozen snapshot of its own policy for a further ~40 M steps. The snapshot is refreshed every 200 training iterations, creating an arms race where the agent must keep improving to stay ahead of its past self.

- **Training script:** `ppo_selfplay.py`
- **Opponent:** Frozen self (numpy MLP snapshot updated every 200 iters)
- **Warm-start:** HU_PPO2 shaped-reward checkpoint (PPO_shaped/checkpoint-2500)
- **Reward:** Sparse ±2 goal signal + 7 dense shaping terms:
  - `+0.010 × Δball_x` — ball moves toward opponent goal (potential-based)
  - `+0.002 × ball_x/17` — ball in attacking half positional bonus
  - `+0.003 × Δdist(p0→ball)` — player 0 chasing ball (potential-based)
  - `+0.003 × Δdist(p1→ball)` — player 1 chasing ball (potential-based)
  - `+0.002 × ball_vx` — ball velocity toward opponent goal
  - `+0.005` — extra bonus when ball is within 5 m of opponent goal
  - `-0.001/step` — stillness penalty when player speed < 0.1 m/s
  - `-0.002` — extra penalty when ball is deep in own half
- **Variation:** `team_vs_policy`, `multiagent=False`
- **Observation:** both players concatenated (336×2 = 672-dim)
- **Action:** MultiDiscrete([3,3,3,3,3,3]) — first 3 for player 0, next 3 for player 1

## Checkpoint

`ray_results/PPO_selfplay/checkpoint-7500`

Trained for 60 M total timesteps (7500 iterations, batch size 8000). Final `episode_reward_mean` ≈ **2.50** (consistently scoring first, rarely conceding).

## Batch script

`scripts/ppo_selfplay.batch`
