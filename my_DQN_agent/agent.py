import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from gym_unity.envs import ActionFlattener
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "DQN"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../ray_results/DQN_1/DQN_Soccer_393f4_00000_0_2026-04-15_17-01-07/checkpoint_000010/checkpoint-10",
)
POLICY_NAME = "default_policy"


class DQNAgent(AgentInterface):
    """
    An agent that loads a DQN policy trained with example_ray_dqn_sp.py
    (team_vs_policy, single_player, flatten_branched).
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Action flattener to convert Discrete action back to MultiDiscrete
        self.flattener = ActionFlattener(env.action_space.nvec)

        # Load configuration from checkpoint file.
        config_dir = os.path.dirname(CHECKPOINT_PATH)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["disable_env_checking"] = True
        config["explore"] = False

        # DQN was trained as single-agent with flatten_branched=True,
        # so we need a simple gym.Env dummy (not MultiAgentEnv)
        obs_space = env.observation_space
        flat_act_space = self.flattener.action_space

        class DummyEnv(gym.Env):
            observation_space = obs_space
            action_space = flat_act_space

            def reset(self):
                return self.observation_space.sample()

            def step(self, action):
                return self.observation_space.sample(), 0.0, True, {}

        tune.registry.register_env("DummyEnv", lambda *_: DummyEnv())
        config["env"] = "DummyEnv"

        # create the Trainer and restore checkpoint
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            action, *_ = self.policy.compute_single_action(
                observation[player_id], explore=False
            )
            # convert Discrete action index back to MultiDiscrete
            actions[player_id] = self.flattener.lookup_action(action)
        return actions
