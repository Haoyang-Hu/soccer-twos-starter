import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results/PPO_shaped/checkpoint-2500",
)
POLICY_NAME = "default_policy"


class HU_PPO2_ShapedRewardAgent(AgentInterface):
    """
    PPO agent trained with ppo_random_reward.py — dense shaped rewards vs random opponent.

    Compared to the baseline (HU_PPO1), this agent adds three dense shaping terms
    on top of the sparse ±2 goal signal to accelerate learning:
      +0.005 × Δball_x   — ball moving toward opponent goal (potential-based)
      +0.002 × ball_x/17 — ball in attacking half positional bonus
      -0.001/step        — stillness penalty when player speed < 0.1 m/s

    Training setup:
    - Training script: ppo_random_reward.py
    - Opponent: random (uniform MultiDiscrete([3,3,3]) per player)
    - Reward: sparse goal signal + shaped reward terms above
    - Variation: team_vs_policy, multiagent=False (both blue players = one agent)
    - Observation: both players concatenated (336×2 = 672-dim)
    - Action: MultiDiscrete([3,3,3,3,3,3]) — first 3 for player 0, next 3 for player 1
    - Trained for 20M timesteps (2500 iterations), final mean reward ~1.6
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "PPO2_shaped_reward"
        ray.init(ignore_reinit_error=True)

        config_dir = os.path.dirname(CHECKPOINT_PATH)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["explore"] = False
        config.get("env_config", {}).pop("opponent_policy", None)

        n_obs = env.observation_space.shape[0]
        team_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs * 2,), dtype=np.float32
        )
        team_act_space = gym.spaces.MultiDiscrete(
            list(env.action_space.nvec) * 2
        )

        class DummyEnv(gym.Env):
            observation_space = team_obs_space
            action_space = team_act_space

            def reset(self):
                return self.observation_space.sample()

            def step(self, _action):
                return self.observation_space.sample(), 0.0, True, {}

        tune.registry.register_env("DummyEnv", lambda *_: DummyEnv())
        config["env"] = "DummyEnv"

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        weights = pickle.loads(ckpt["worker"])["state"][POLICY_NAME]
        weights.pop("_optimizer_variables", None)
        self.policy = agent.get_policy(POLICY_NAME)
        self.policy.set_weights(weights)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        combined_obs = np.concatenate([observation[0], observation[1]])
        team_action, *_ = self.policy.compute_single_action(combined_obs, explore=False)
        n = len(team_action) // 2
        return {0: team_action[:n], 1: team_action[n:]}
