# PPO training: team controller vs random opponent
#
# - Env variation: team_vs_policy (your agent controls both players on one team)
# - Algorithm: PPO with MultiDiscrete action space
# - Multiagent: No — the team is treated as a single agent by RLlib
# - Opponent: random policy (takes random actions every step)
# - Compared to ppo_sp_still: harder task — opponent now moves unpredictably,
#   forcing the agent to learn to handle a non-trivial adversary
#   Trained

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 4  # original
# NUM_ENVS_PER_WORKER = 2  # tuned for 6C/12T, 16GB RAM, GTX 1660 Ti 6GB


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_1",
        config={
            # system settings
            # original: "num_gpus": 1, "num_workers": 8, "num_envs_per_worker": 5
            # note: num_gpus=0 — Ray 1.4 + torch has a bug where workers crash
            # with IndexError at torch_policy.py:155 when num_gpus>0 on the trainer.
            # CPU training is fine here since the MLP is small and Unity sim is the bottleneck.
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
            # original: defaults (rollout_fragment_length=200, train_batch_size=4000)
            "rollout_fragment_length": 500,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
        },
        stop={
            "timesteps_total": 20_000_000, #20000000,  # 15M
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
