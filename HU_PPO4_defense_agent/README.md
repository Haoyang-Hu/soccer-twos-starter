# HU PPO4 Defense Agent

**Agent name:** PPO4_defense

## Description

PPO agent trained with `ppo_selfplay_defense.py` — frozen self-play with the full offensive reward suite from HU_PPO3 **plus three new defensive terms** that only activate when the ball is in the agent's own half. The goal is to preserve HU_PPO3's strong offense (avg 12.8 goals vs CEIA) while reducing goals conceded (avg 6.8 → ideally < 3).

- **Training script:** `ppo_selfplay_defense.py`
- **Opponent:** Frozen self (numpy MLP snapshot updated every 200 iters)
- **Warm-start:** HU_PPO3 self-play checkpoint (`PPO_selfplay/checkpoint-7500`, 60 M steps, mean ~2.50)

## Reward Terms

**Offensive (terms 1–7, unchanged from HU_PPO3):**
- `+0.010 × Δball_x` — ball toward opponent goal (potential-based)
- `+0.002 × ball_x/17` — attacking-half zone bonus
- `+0.003 × Δdist(p0→ball)` — player 0 chasing ball (potential-based)
- `-0.001/step` — stillness penalty (speed < 0.1 m/s)
- `+0.002 × max(0, ball_vx)` — ball velocity toward opponent goal
- `+0.005 × max(0,(ball_x−12)/5)` — goal-zone amplifier (final 5 m)
- `-0.002 × max(0,(−ball_x−10)/7)` — danger-zone penalty

**Defensive (terms 8–10, new — only when ball_x < 0):**
- `+0.002` — goal-side positioning: player between ball and own goal
- `+0.003 × max(0, ball_vx)` — clearance bonus: ball moving away from own goal
- `+0.002 × max(0, 1 − dist/10)` — defensive proximity: closing down ball-holder

## Checkpoint

`ray_results/PPO_selfplay_defense/checkpoint-12500`

Trained for 100 M total timesteps (12500 iterations, 40 M new steps on top of HU_PPO3). Final `episode_reward_mean` ≈ **2.53**.

## Batch script

`scripts/ppo_selfplay_defense.batch`
