# HU_PPO3_selfplay_agent/agent.py
#
# Loads a PPO policy trained with ppo_selfplay.py and exposes it as an
# AgentInterface for soccer-twos evaluation and competition.
#
# ============================================================================
# TRAINING OVERVIEW
# ============================================================================
# Script:      ppo_selfplay.py
# Warm-start:  HU_PPO2 shaped-reward checkpoint (PPO_shaped/checkpoint-2500,
#              trained ~20 M steps vs random opponent)
# Opponent:    Frozen self — a pure-numpy snapshot of the current policy
#              deployed to all workers and refreshed every 200 training iters.
#              This creates an arms race: the agent must keep improving to beat
#              its own recent past, preventing the plateau that occurs with a
#              fixed random opponent.
# Total steps: 60 M timesteps (7500 iterations @ train_batch_size=8000)
# Final mean:  episode_reward_mean ≈ 2.50  (near the +2 single-goal ceiling,
#              with shaping adding up to ~0.5 per episode)
#
# ============================================================================
# AGENT STRUCTURE  (team_vs_policy, multiagent=False)
# ============================================================================
# Both blue players are controlled as a SINGLE RLlib agent via
# TeamVsPolicyWrapper.  The wrapper concatenates the two players' observations
# and expects a joint action covering both:
#
#   observation_space : Box(672,)            — 336-dim per player, concatenated
#   action_space      : MultiDiscrete([3,3,3,3,3,3])
#                         indices [0:3] → player 0 (forward/back, strafe, rotate)
#                         indices [3:6] → player 1
#
# act() concatenates observation[0] and observation[1] → 672-dim vector,
# runs a single forward pass, then splits the 6-dim joint action back into
# per-player dicts: {0: action[:3], 1: action[3:]}.
#
# ============================================================================
# REWARD SHAPING  (7 dense terms on top of sparse ±2 goal signal)
# ============================================================================
#   +0.010 × Δball_x           ball moves toward opponent goal  (potential)
#   +0.002 × ball_x / 17       zone bonus when ball in attacking half
#   +0.003 × Δdist(p0 → ball)  player 0 chasing ball           (potential)
#   +0.003 × Δdist(p1 → ball)  player 1 chasing ball           (potential)
#   +0.002 × ball_vx            ball velocity toward opponent goal
#   +0.005                      extra bonus within 5 m of opponent goal line
#   -0.001 / step               stillness penalty  (speed < 0.1 m/s)
#   -0.002                      danger-zone penalty when ball is deep in own half
#
# ============================================================================
# CHECKPOINT LOADING
# ============================================================================
# Ray 1.13 stores checkpoint weights at ckpt["worker"]["state"][policy_name],
# NOT at the "weights" key that agent.restore() expects.  We therefore load
# the pickle manually and call policy.set_weights() directly — the same
# pattern used in ceia_baseline_agent/agent_ray.py.
#
# A DummyEnv is registered to give RLlib the correct observation/action spaces
# without spawning a real Unity binary during evaluation.

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
    "ray_results/PPO_selfplay/checkpoint-7500",
)
POLICY_NAME = "default_policy"


class PPO3SelfPlayAgent(AgentInterface):

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "PPO3_selfplay"
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
