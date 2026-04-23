# PPO4 Defense Agent

**Agent name:** PPO4_defense

## Overview

This is the **final submitted agent**. It is the fourth and most refined policy in an iterative training pipeline:

```
PPO1 (baseline, sparse reward, vs random)
  └─> PPO2 (dense reward shaping, vs random)
        └─> PPO3 (frozen self-play, stronger shaping)
              └─> PPO4 (self-play + defensive terms)  ← submitted
```

---

## Training Pipeline

### PPO1 — Sparse Baseline (not submitted)

Vanilla PPO trained against a random opponent with only the sparse ±2 goal reward signal. Used as a sanity check to confirm the environment works and the agent can learn to score at all. Reached `episode_reward_mean` ≈ **0.8** after 10 M steps.

---

### PPO2 — Dense Reward Shaping vs Random

**Script:** `ppo_random_reward.py`  
**Opponent:** Random (uniform MultiDiscrete([3,3,3]) per player)  
**Checkpoint:** `ray_results/PPO_shaped/checkpoint-2500`  
**Training:** 20 M timesteps (2500 iterations, batch size 8000)  
**Final reward mean:** ≈ **1.61** (near the +2 single-goal ceiling)

PPO2 added three dense shaping terms on top of the sparse goal signal to teach the agent to push the ball toward the opponent goal, stay in the attacking half, and avoid idling:

| Term | Formula | Purpose |
|------|---------|---------|
| Ball progress | `+0.005 × Δball_x` | Potential-based reward for moving ball toward opponent goal |
| Attacking-half bonus | `+0.002 × ball_x/17` | Positional incentive to keep ball in opponent half |
| Stillness penalty | `−0.001/step` | Discourages stationary players (speed < 0.1 m/s) |

- **Variation:** `team_vs_policy`, `multiagent=False`
- **Observation:** both players concatenated (336 × 2 = 672-dim)
- **Action:** MultiDiscrete([3,3,3,3,3,3]) — first 3 dims for player 0, next 3 for player 1

Compared to PPO1, PPO2 learned to score reliably against random opponents and developed consistent ball-chasing behavior, providing a much stronger warm-start for self-play.

---

### PPO3 — Frozen Self-Play with 7-Term Shaping

**Script:** `ppo_selfplay.py`  
**Opponent:** Frozen self (numpy MLP snapshot refreshed every 200 training iterations)  
**Warm-start:** PPO2 checkpoint (`PPO_shaped/checkpoint-2500`)  
**Checkpoint:** `ray_results/PPO_selfplay/checkpoint-7500`  
**Training:** 60 M total timesteps (7500 iterations, ~40 M new steps on top of PPO2)  
**Final reward mean:** ≈ **2.50** (consistently scoring first, rarely conceding)

PPO3 replaced the random opponent with a frozen snapshot of the agent's own past policy. Every 200 iterations the snapshot is refreshed, creating a continual arms race. The reward scheme was expanded to 7 dense terms:

| Term | Formula | Purpose |
|------|---------|---------|
| Ball progress (p-based) | `+0.010 × Δball_x` | Stronger potential shaping toward opponent goal |
| Attacking-half bonus | `+0.002 × ball_x/17` | Ball in opponent half |
| P0 chases ball | `+0.003 × Δdist(p0→ball)` | Keeps player 0 close to ball |
| P1 chases ball | `+0.003 × Δdist(p1→ball)` | Keeps player 1 close to ball |
| Ball velocity | `+0.002 × ball_vx` | Rewards ball moving toward opponent goal |
| Goal-zone bonus | `+0.005` when ball within 5 m of opponent goal | Incentivizes final-third play |
| Stillness penalty | `−0.001/step` | Discourages idle players |
| Danger-zone penalty | `−0.002` when ball is deep in own half | Discourages conceding territory |

- **Variation:** `team_vs_policy`, `multiagent=False`
- **Observation:** 672-dim (same as PPO2)
- **Action:** MultiDiscrete([3,3,3,3,3,3])

PPO3 scored an average of **12.8 goals vs CEIA** baseline but conceded an average of **6.8 goals**, motivating the defensive extension in PPO4.

---

## PPO4 — Self-Play + Defensive Terms (Final Submission)

**Script:** `ppo_selfplay_defense.py`  
**Opponent:** Frozen self (numpy MLP snapshot refreshed every 200 iterations)  
**Warm-start:** PPO3 checkpoint (`PPO_selfplay/checkpoint-7500`, 60 M steps)  
**Checkpoint:** `ray_results/PPO_selfplay_defense/checkpoint-12500`  
**Training:** 100 M total timesteps (12500 iterations, 40 M new steps on top of PPO3)  
**Final reward mean:** ≈ **2.53**

PPO4 keeps all 7 offensive terms from PPO3 unchanged and adds 3 new defensive terms that only activate when the ball is in the agent's own half (`ball_x < 0`). The goal is to preserve PPO3's strong offense while reducing goals conceded.

### Reward Terms

**Offensive (terms 1–7, identical to PPO3):**

| # | Term | Formula |
|---|------|---------|
| 1 | Ball progress | `+0.010 × Δball_x` |
| 2 | Attacking-half bonus | `+0.002 × ball_x/17` |
| 3 | P0 chases ball | `+0.003 × Δdist(p0→ball)` |
| 4 | Stillness penalty | `−0.001/step` |
| 5 | Ball velocity | `+0.002 × max(0, ball_vx)` |
| 6 | Goal-zone amplifier | `+0.005 × max(0, (ball_x − 12) / 5)` — final 5 m |
| 7 | Danger-zone penalty | `−0.002 × max(0, (−ball_x − 10) / 7)` |

**Defensive (terms 8–10, new — only active when `ball_x < 0`):**

| # | Term | Formula | Purpose |
|---|------|---------|---------|
| 8 | Goal-side positioning | `+0.002` when player is between ball and own goal | Teaches defensive positioning |
| 9 | Clearance bonus | `+0.003 × max(0, ball_vx)` | Rewards kicking ball away from own goal |
| 10 | Defensive proximity | `+0.002 × max(0, 1 − dist/10)` | Rewards closing down the ball-holder |

### Batch script

`scripts/ppo_selfplay_defense.batch`
