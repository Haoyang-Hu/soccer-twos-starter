# HU PPO1 Agent

**Agent name:** PPO1

## Description

An agent using a PPO policy trained with `example_ray_ppo_sp_still.py` on the `team_vs_policy` variation with `single_player=True` and `flatten_branched=True`. The opponent is fixed (always takes action 0). Loads weights from `ray_results/PPO_1/`.

## Checkpoint

`ray_results/PPO_1/PPO_Soccer_1ae3c_00000_0_2026-04-15_20-20-43/checkpoint_001000/`

Trained for 1000 checkpoints with a batch size of 12,000 and rollout fragment length of 500.
