# PPO training with shaped rewards vs random opponent
#
# ============================================================================
# AGENT STRUCTURE: one team = one agent (NOT one player = one agent)
# ============================================================================
# EnvType.team_vs_policy + multiagent=False routes through TeamVsPolicyWrapper
# (soccer_twos/wrappers.py:448), which presents the two blue players as a
# single RLlib agent:
#
#   observation_space = Box(shape=(672,))           # 336 (player 0) ⊕ 336 (player 1)
#   action_space      = MultiDiscrete([3,3,3,3,3,3]) # 3 branches × 2 players
#
# On each step the wrapper splits the 6-dim joint action: indices [0:3] go to
# player 0, indices [3:6] go to player 1. Both teammates share the same policy
# weights and observe each other's state, so coordination is learned implicitly
# through the shared encoder rather than via emergent multi-agent communication.
#
# The orange team is driven by `opponent_policy` (random here, see env_config
# below) and is NOT learned. RLlib only sees one policy ("default_policy")
# producing one joint action per step.
#
# (For per-player learning you would use EnvType.multiagent_player with
# multiagent=True, which exposes RLlib's MultiAgentEnv interface — not used here.)
#
# ============================================================================
# BASE REWARD (from Unity, via TeamVsPolicyWrapper)
# ============================================================================
# Each player receives +1 when their team scores, -1 when the opponent scores,
# 0 otherwise. TeamVsPolicyWrapper sums both players: total base = +2 / -2 / 0.
# This signal is purely sparse — only fires at goal events, which can be tens
# of seconds apart, making credit assignment hard without shaping.
#
# ============================================================================
# SHAPED REWARD (added on top of base each step, see RewardShaperWrapper)
# ============================================================================
#   1. Ball-toward-goal  (+0.005 × Δball_x): potential-based dense reward for
#                         moving the ball toward the opponent goal. Because
#                         it is potential-based (F = Φ(s') − Φ(s)) it does not
#                         change the optimal policy, only learning speed.
#   2. Ball-zone         (+0.002 × ball_x/17): continuous bonus when ball is
#                         in opponent half, penalty in own half. Encourages
#                         possession in the attacking third.
#   3. Stillness penalty (-0.001/step): discourages player 0 from idling
#                         (speed < 0.1 m/s). Counters a degenerate strategy
#                         where standing still gives 0 reward, which is locally
#                         optimal under sparse goal feedback.
#
#   Scale: shaping maxes at ~0.007/step vs ±2 for a goal → ~0.35% per step,
#   small enough not to dominate the true objective but dense enough to guide
#   exploration in the first ~1M steps before the agent starts scoring.
#
# Coordinates (Soccer-Twos): position is 2-D (x, z) on the ground plane;
# blue attacks toward +x, orange toward −x; field ≈ x ∈ [-17, 17], z ∈ [-7, 7].
#
# Shaping requires `info["ball_info"]` and `info["player_info"]`, which the
# binary only emits when sending the 345-dim per-player obs (training env);
# when missing (e.g. on reset) shaping is silently skipped.
#
# ============================================================================
# OPPONENT: random
# ============================================================================
# This file isolates the random-opponent stage to verify steady training
# end-to-end. Matches example_ray_team_vs_random.py, which trained
# successfully on PACE.
#
# soccer_twos defaults `opponent_policy` to `env.action_space.sample()` when
# the key is omitted from env_config (wrappers.py:495), so we don't pass it
# explicitly.
#
# ============================================================================
# RLlib config notes
# ============================================================================
# - num_gpus=0: Ray 1.13 + torch has a bug where workers crash with
#   IndexError at torch_policy.py:155 when num_gpus>0 on the trainer.
#   The MLP is small and the Unity sim is the bottleneck, so CPU is fine.
# - num_workers=16: each worker spawns its own Unity binary, so memory and
#   CPU scale linearly. Reduce on smaller machines.
# - train_batch_size=8000, rollout_fragment_length=500 → 16 workers × 500 = 8000
#   per training step (1 SGD round of 10 epochs over 8k samples).

import numpy as np
import gym
import ray
from ray import tune
import soccer_twos
from soccer_twos import EnvType


NUM_ENVS_PER_WORKER = 1


# ---------------------------------------------------------------------------
# Reward-shaping wrapper
# ---------------------------------------------------------------------------
# Soccer-Twos coordinate system (from the binary):
#   - position is 2-D (x, z on the ground plane)
#   - Blue team attacks toward +x, Orange team attacks toward −x
#   - The field is roughly x ∈ [-17, 17], z ∈ [-7, 7]

FIELD_HALF_X = 17.0   # half-length of the pitch (used to normalise ball position)

# Reward coefficients (small relative to the ±2 goal signal)
COEFF_BALL_TOWARD_GOAL = 0.005   # per-step for Δx toward opponent goal
COEFF_BALL_ZONE = 0.002          # per-step positional bonus / penalty
COEFF_STILLNESS = 0.001          # per-step penalty when velocity ≈ 0


class RewardShaperWrapper(gym.core.Wrapper):
    """
    Wraps a TeamVsPolicyWrapper env and adds dense reward shaping.

    Relies on the info dict containing 'player_info' and 'ball_info'
    (present when the compiled binary sends 345-dim obs per player).
    When the info is missing (e.g. first step), shaping is skipped.
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_ball_x = None

    def reset(self):
        obs = self.env.reset()
        self._prev_ball_x = None
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)

        # base_reward = reward[0] + reward[1] from TeamVsPolicyWrapper
        # = +2 on your goal, -2 on opponent goal, 0 otherwise (sparse)
        shaping = 0.0

        # info comes from TeamVsPolicyWrapper which passes info[0]
        # (player 0's dict). Contains 'player_info' and 'ball_info' only
        # when the binary env sends 345-dim obs (training env, not watch env).
        if isinstance(info, dict) and "ball_info" in info:
            ball_x = float(info["ball_info"]["position"][0])
            # ball_x ∈ [-17, +17]: +17 = opponent goal, -17 = own goal

            # --- 1. Ball-toward-goal: potential-based Δx reward ---
            # Rewards every step the ball moves in the +x direction (toward opponent goal).
            # Penalises every step the ball moves in the -x direction (toward own goal).
            # Being potential-based (F = Φ(s') - Φ(s)) means it does not change the
            # optimal policy, only the speed of learning.
            if self._prev_ball_x is not None:
                delta_x = ball_x - self._prev_ball_x
                shaping += COEFF_BALL_TOWARD_GOAL * delta_x
            self._prev_ball_x = ball_x

            # --- 2. Ball-zone: continuous positional bonus/penalty ---
            # norm_x ∈ [-1, +1].  Gives a steady bonus when ball is in opponent half
            # and a steady penalty when in own half.  Treats the field as a rectangle
            # where x = 0 is midfield, x = +FIELD_HALF_X is the opponent goal line.
            norm_x = np.clip(ball_x / FIELD_HALF_X, -1.0, 1.0)
            shaping += COEFF_BALL_ZONE * norm_x

            # --- 3. Stillness penalty ---
            # Penalises player 0 for standing still (speed < 0.1 m/s).
            # Prevents the policy from learning to idle when the opponent misses.
            if "player_info" in info:
                vel = info["player_info"]["velocity"]  # [vx, vz] in world space
                speed = np.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2)
                if speed < 0.1:
                    shaping -= COEFF_STILLNESS

        # total reward = sparse goal signal + dense shaping
        # shaping magnitude: max ~0.007/step vs ±2 for a goal
        return obs, base_reward + shaping, done, info


# ---------------------------------------------------------------------------
# Custom env creator
# ---------------------------------------------------------------------------

def create_shaped_env(env_config: dict = {}):
    """
    Builds: MultiAgentUnityWrapper → TeamVsPolicyWrapper
              → EnvChannelWrapper → RewardShaperWrapper
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = RewardShaperWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("SoccerShaped", create_shaped_env)

    analysis = tune.run(
        "PPO",
        name="PPO_shaped",
        config={
            # num_gpus=0: Ray 1.13+torch has a bug where workers crash with
            # IndexError at torch_policy.py:155 when num_gpus>0 on the trainer.
            "num_gpus": 0,
            "num_workers": 16,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "SoccerShaped",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                # Omit opponent_policy → soccer_twos defaults to random opponent
                # (matches example_ray_team_vs_random.py which trained successfully)
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 8000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
        },
        stop={
            "timesteps_total": 20_000_000,
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
