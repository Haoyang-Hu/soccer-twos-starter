# DQN training: single player vs still opponent
#
# - Env variation: team_vs_policy (your agent controls 1 player; opponent does nothing)
# - Algorithm: DQN (requires flattened Discrete action space via flatten_branched=True)
# - Multiagent: No — env appears as a standard single-agent gym env to RLlib
# - Opponent: fixed, always takes action 0 (does nothing)
# - Good starting point: simplest possible setup to get a policy learning to score

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1 # 5


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "DQN",
        name="DQN_1",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 2,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "flatten_branched": True,
                "single_player": True,
            },
            "model": {
                "fcnet_hiddens": [512, 256],
            },
        },
        stop={
            "timesteps_total": 10000, #20000000,  # 20M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
