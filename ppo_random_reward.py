# PPO training with shaped rewards and opponent curriculum
#
# BASE REWARD (from Unity, via TeamVsPolicyWrapper):
#   Each player receives +1 when their team scores, -1 when the opponent scores, 0 otherwise.
#   TeamVsPolicyWrapper sums both players: total base = +2 / -2 / 0.
#   This signal is purely sparse — only fires at goal events.
#
# SHAPED REWARD (added on top of base each step):
#   1. Ball-toward-goal  (+0.005 × Δball_x): dense reward for moving ball toward opponent goal
#   2. Ball-zone         (+0.002 × ball_x/17): continuous bonus when ball is in opponent half,
#                         penalty when in own half — treats the field as a rectangle where
#                         x = +17 is opponent goal and x = -17 is own goal
#   3. Stillness penalty (-0.001/step): discourages standing still
#
#   Scale: shaping maxes at ~0.007/step vs ±2 for a goal → ~0.35% per step,
#   small enough not to dominate but dense enough to guide exploration.
#
# OPPONENT CURRICULUM (staged difficulty):
#   Stage 0: still opponent      — [0,0,0] = no movement, easiest baseline
#   Stage 1: random opponent     — promotes at episode_reward_mean > 0.5
#   Stage 2: frozen-self         — promotes at episode_reward_mean > 1.0,
#                                   uses a snapshot of the current policy weights
#
# Env variation: team_vs_policy (agent controls both players on blue team)
# Action space: MultiDiscrete([3,3,3,3,3,3]) — no flattening
# Observation: 672 (336 per player, concatenated)

import copy

import numpy as np
import torch
import gym
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import soccer_twos
from soccer_twos import EnvType


NUM_ENVS_PER_WORKER = 2


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
# Opponent curriculum callback
# ---------------------------------------------------------------------------
# Stage 0: still opponent   — [0,0,0] per player
# Stage 1: random opponent  — env.action_space.sample() per player
# Stage 2: frozen self      — snapshot of current policy weights
#
# Note on obs dimensions:
#   The trained policy takes 672-dim (two players concatenated).
#   opponent_policy is called per-player with 336-dim obs.
#   For the frozen-self opponent we tile the obs to fake a 672-dim input
#   and use only the first player's logits (first 9 of 18 outputs).

STAGE_THRESHOLDS = [0.5, 1.0]  # promote 0→1 at 0.5, 1→2 at 1.0

# Per-player action branches (MultiDiscrete([3,3,3]))
_SINGLE_PLAYER_BRANCHES = [3, 3, 3]


def _find_tvp(env):
    """Walk the gym wrapper chain to find the TeamVsPolicyWrapper."""
    while hasattr(env, "env"):
        if hasattr(env, "set_opponent_policy"):
            return env
        env = env.env
    return env


def _make_frozen_opponent(model):
    """
    Returns a callable that acts as a frozen-self opponent for a single player.

    The trained policy takes 672-dim team obs and outputs 18 logits
    (MultiDiscrete [3,3,3,3,3,3]). opponent_policy is called per player
    with 336-dim obs, so we tile it to 672 and read the first player's
    action (first 9 logits → branches [3,3,3]).
    """
    frozen = copy.deepcopy(model)
    frozen.eval()

    def opponent(obs, _frozen=frozen):
        # tile single-player obs to match the team input the policy was trained on
        team_obs = np.tile(obs, 2).astype(np.float32)  # 336 → 672
        obs_t = torch.from_numpy(team_obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = _frozen({"obs": obs_t}, [], None)
        logits_np = logits.cpu().numpy()[0]  # shape [18]
        # decode first player's branches from first 9 logits
        action = []
        offset = 0
        for n in _SINGLE_PLAYER_BRANCHES:
            action.append(int(np.argmax(logits_np[offset:offset + n])))
            offset += n
        return action

    return opponent


class OpponentCurriculumCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self._stage = 0

    def on_train_result(self, *, trainer, result, **_):
        mean_reward = result["episode_reward_mean"]

        # Guard: NaN means no episodes have completed yet (nan <= x is False
        # in IEEE 754, so without this check NaN would slip past the threshold
        # guard and trigger a premature curriculum promotion).
        if np.isnan(mean_reward):
            return

        if self._stage >= len(STAGE_THRESHOLDS):
            return
        if mean_reward <= STAGE_THRESHOLDS[self._stage]:
            return

        self._stage += 1
        stage_name = ["still", "random", "frozen-self"][self._stage]
        print(f"==== CURRICULUM: promoting to stage {self._stage} ({stage_name}) "
              f"at reward {mean_reward:.2f} ====")

        if self._stage == 1:
            # Switch remote workers to random opponent
            def _set_random(worker):
                for env in worker.async_env.get_unwrapped():
                    tvp = _find_tvp(env)
                    # default-arg captures tvp correctly per loop iteration
                    tvp.set_opponent_policy(
                        lambda _obs, _tvp=tvp: _tvp.env.action_space.sample()
                    )

            for w in trainer.workers.remote_workers():
                w.apply.remote(_set_random)

        elif self._stage == 2:
            # Snapshot current policy weights into a frozen model copy
            def _set_frozen_self(worker):
                local_policy = worker.policy_map["default_policy"]
                frozen_fn = _make_frozen_opponent(local_policy.model)
                for env in worker.async_env.get_unwrapped():
                    tvp = _find_tvp(env)
                    tvp.set_opponent_policy(frozen_fn)

            for w in trainer.workers.remote_workers():
                w.apply.remote(_set_frozen_self)


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
    ray.init()

    tune.registry.register_env("SoccerShaped", create_shaped_env)

    analysis = tune.run(
        "PPO",
        name="PPO_shaped",
        config={
            # num_gpus=0: Ray 1.13+torch has a bug where workers crash with
            # IndexError at torch_policy.py:155 when num_gpus>0 on the trainer.
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # "disable_env_checking": True,  # not supported in Ray 1.4
            "callbacks": OpponentCurriculumCallback,
            # RL setup
            "env": "SoccerShaped",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                # Stage 0: still opponent — MultiDiscrete([3,3,3]) needs an array
                "opponent_policy": lambda *_: [0, 0, 0],
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
        },
        stop={
            "timesteps_total": 4_000_000,
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
