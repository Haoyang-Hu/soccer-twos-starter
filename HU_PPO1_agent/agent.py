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
    "ray_results/PPO_1/checkpoint_005000/checkpoint-5000",
)
POLICY_NAME = "default_policy"


class PPO1Agent(AgentInterface):
    """
    An agent that loads a PPO policy trained with example_ray_ppo_sp_still.py
    (team_vs_policy, single_player=False, flatten_branched=False).

    The policy controls both team members as a single agent:
    - observation: both players' obs concatenated (336*2 = 672)
    - action: MultiDiscrete([3,3,3,3,3,3]) — first 3 for player 0, next 3 for player 1
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

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
        config["explore"] = False
        # remove unpicklable/unneeded opponent_policy lambda from env_config
        config.get("env_config", {}).pop("opponent_policy", None)

        # The policy was trained as a team agent (single_player=False, flatten_branched=False):
        # observation = both players concatenated (336*2=672), action = MultiDiscrete for both players
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

        # create the Trainer and manually load weights from checkpoint
        # (agent.restore() fails because the checkpoint format stores weights under
        # ["worker"]["state"][policy_name], not the "weights" key restore() expects)
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        weights = pickle.loads(ckpt["worker"])["state"][POLICY_NAME]
        weights.pop("_optimizer_variables", None)
        self.policy = agent.get_policy(POLICY_NAME)
        self.policy.set_weights(weights)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        # Concatenate both players' observations — the policy was trained on the full team obs
        combined_obs = np.concatenate([observation[0], observation[1]])
        team_action, *_ = self.policy.compute_single_action(combined_obs, explore=False)
        # Split the 6-element action array back into per-player MultiDiscrete actions
        n = len(team_action) // 2
        return {0: team_action[:n], 1: team_action[n:]}
