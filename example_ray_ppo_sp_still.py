# PPO training: single player vs still opponent
#
# - Env variation: team_vs_policy (your agent controls 1 player; opponent does nothing)
# - Algorithm: PPO with MultiDiscrete action space (no flattening needed unlike DQN)
# - Multiagent: No — env appears as a standard single-agent gym env to RLlib
# - Opponent: fixed, always takes action 0 (does nothing)
# - Compared to example_ray_dqn_sp.py: same setup but uses PPO instead of DQN,
#   with a larger batch size and explicit rollout_fragment_length for stability

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_SP",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 20000000,  # 15M
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
