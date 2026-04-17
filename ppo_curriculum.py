# PPO training with 4-stage curriculum opponent
#
# ============================================================================
# CURRICULUM STAGES
# ============================================================================
#   0. Still      — opponent always returns [0,0,0] (no movement)
#                   Promotes when episode_reward_mean > THRESH_STILL (1.0)
#
#   1. Random     — opponent samples uniformly from MultiDiscrete([3,3,3])
#                   Promotes when episode_reward_mean > THRESH_RANDOM (0.3)
#
#   2. Frozen self — numpy snapshot of the current policy weights taken at
#                   promotion from stage 1, then refreshed every
#                   FROZEN_UPDATE_INTERVAL training iterations.
#                   Opponent receives 336-dim per-player obs; we tile it to
#                   672 to match the team-policy input, then take the first 3
#                   branches of the 6-branch output as the single player action.
#                   Promotes when episode_reward_mean > THRESH_FROZEN (0.2)
#                   OR after FROZEN_STAGE_MAX_ITERS iterations.
#
#   3. CEIA       — fixed opponent loaded from ceia_baseline_agent checkpoint
#                   (256×256 MLP, 336-dim input, 9 output logits = [3,3,3]).
#                   Final stage; no further promotion.
#
# ============================================================================
# REWARD (same shaped reward as ppo_random_reward.py)
# ============================================================================
#   base: +2 / -2 / 0 from TeamVsPolicyWrapper (goal scored / conceded / else)
#   shaping:
#     +0.005 × Δball_x  (potential-based; ball moves toward opponent goal)
#     +0.002 × ball_x/17  (zone bonus: ball in attacking half)
#     -0.001/step if player 0 speed < 0.1  (stillness penalty)
#
# ============================================================================
# IMPLEMENTATION NOTES
# ============================================================================
# - All opponent-update functions are defined at module level so Ray can
#   pickle them for remote worker calls.
# - NaN guard in on_train_result prevents premature stage promotion on the
#   first training iteration (before any episodes complete).
# - Workers are updated via trainer.workers.remote_workers() + w.apply.remote()
#   to avoid the local-worker async_env=None crash.
# - set_opponent_policy() is a built-in method on TeamVsPolicyWrapper.

import os
import pickle
import numpy as np
import gym
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

# ---------------------------------------------------------------------------
# Curriculum config
# ---------------------------------------------------------------------------

THRESH_STILL  = 1.0   # episode_reward_mean to promote stage 0 → 1
THRESH_RANDOM = 0.3   # episode_reward_mean to promote stage 1 → 2
THRESH_FROZEN = 0.2   # episode_reward_mean to promote stage 2 → 3
FROZEN_UPDATE_INTERVAL = 200   # re-snapshot frozen self every N iters in stage 2
FROZEN_STAGE_MAX_ITERS = 800   # force-promote to CEIA after this many iters in stage 2

CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos"
    "/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02"
    "/checkpoint_002449/checkpoint-2449",
)
CEIA_POLICY_NAME = "default"

NUM_ENVS_PER_WORKER = 1
FIELD_HALF_X = 17.0
COEFF_BALL_TOWARD_GOAL = 0.005
COEFF_BALL_ZONE        = 0.002
COEFF_STILLNESS        = 0.001


# ---------------------------------------------------------------------------
# Pure-numpy MLP — picklable, no Ray/torch required at inference time
# ---------------------------------------------------------------------------

class NumpyMLP:
    """
    Reconstruct an RLlib FCNet as a pure-numpy forward pass.
    Weight keys follow the RLlib naming convention:
      _hidden_layers.{i}._model.0.weight / .bias
      _logits._model.0.weight / .bias
    """

    def __init__(self, state_dict, output_branches):
        self.layers = []
        i = 0
        while f"_hidden_layers.{i}._model.0.weight" in state_dict:
            W = np.array(state_dict[f"_hidden_layers.{i}._model.0.weight"], dtype=np.float32)
            b = np.array(state_dict[f"_hidden_layers.{i}._model.0.bias"],   dtype=np.float32)
            self.layers.append((W, b))
            i += 1
        self.W_out = np.array(state_dict["_logits._model.0.weight"], dtype=np.float32)
        self.b_out = np.array(state_dict["_logits._model.0.bias"],   dtype=np.float32)
        self.output_branches = output_branches

    def act(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for W, b in self.layers:
            x = np.maximum(0.0, x @ W.T + b)
        logits = x @ self.W_out.T + self.b_out
        action, offset = [], 0
        for n in self.output_branches:
            action.append(int(np.argmax(logits[offset: offset + n])))
            offset += n
        return np.array(action, dtype=np.int64)


# ---------------------------------------------------------------------------
# Opponent factory functions
# ---------------------------------------------------------------------------

def _still_opponent(obs):
    return np.zeros(3, dtype=np.int64)


def _random_opponent(obs):
    return np.array([np.random.randint(3) for _ in range(3)], dtype=np.int64)


def _make_frozen_opponent(state_dict):
    """
    Create an opponent from the current team-policy weights (672-dim input,
    18 output logits = [3,3,3,3,3,3]).  When called as an orange-team
    opponent it receives 336-dim per-player obs, which we tile to 672.
    We then use only the first 3 action branches (for one player).
    """
    mlp = NumpyMLP(state_dict, output_branches=[3, 3, 3, 3, 3, 3])

    def opponent(obs_336):
        team_obs = np.tile(obs_336.astype(np.float32), 2)
        team_action = mlp.act(team_obs)
        return team_action[:3]

    return opponent


def _load_ceia_opponent():
    """Load CEIA checkpoint as a per-player opponent (336-dim → [3,3,3])."""
    with open(CEIA_CHECKPOINT, "rb") as f:
        ckpt = pickle.load(f)
    state_dict = pickle.loads(ckpt["worker"])["state"][CEIA_POLICY_NAME]
    state_dict.pop("_optimizer_variables", None)
    mlp = NumpyMLP(state_dict, output_branches=[3, 3, 3])

    def opponent(obs_336):
        return mlp.act(obs_336.astype(np.float32))

    return opponent


# ---------------------------------------------------------------------------
# Worker update (module-level → picklable for Ray remote calls)
# ---------------------------------------------------------------------------

def _apply_opponent_to_worker(worker, opponent_fn):
    """Set opponent_policy on all TeamVsPolicyWrappers in this worker."""
    for env in worker.async_env.get_sub_environments():
        e = env
        while e is not None:
            if hasattr(e, "set_opponent_policy"):
                e.set_opponent_policy(opponent_fn)
                break
            e = getattr(e, "env", None)


def _update_all_workers(trainer, opponent_fn):
    for w in trainer.workers.remote_workers():
        w.apply.remote(_apply_opponent_to_worker, opponent_fn)


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self._stage = 0
        self._iters_in_stage = 0

    def on_train_result(self, trainer, result, **kwargs):
        mean = result.get("episode_reward_mean", float("nan"))
        if np.isnan(mean):
            return

        stage = self._stage
        iters  = self._iters_in_stage
        self._iters_in_stage += 1

        promote = False

        if stage == 0 and mean > THRESH_STILL:
            promote = True

        elif stage == 1 and mean > THRESH_RANDOM:
            promote = True

        elif stage == 2:
            if mean > THRESH_FROZEN or iters >= FROZEN_STAGE_MAX_ITERS:
                promote = True
            elif iters > 0 and iters % FROZEN_UPDATE_INTERVAL == 0:
                # Refresh frozen-self snapshot with latest weights
                state_dict = trainer.get_policy("default_policy").get_weights()
                opp = _make_frozen_opponent(state_dict)
                _update_all_workers(trainer, opp)
                print(f"[Curriculum] Stage 2: refreshed frozen-self snapshot "
                      f"(iter {iters}, mean={mean:.3f})")

        if promote and stage < 3:
            self._stage += 1
            self._iters_in_stage = 0
            self._transition(trainer, self._stage, mean)

    def _transition(self, trainer, new_stage, mean):
        if new_stage == 1:
            opp = _random_opponent
            label = "random"

        elif new_stage == 2:
            state_dict = trainer.get_policy("default_policy").get_weights()
            opp = _make_frozen_opponent(state_dict)
            label = "frozen-self"

        elif new_stage == 3:
            if not os.path.exists(CEIA_CHECKPOINT):
                print("[Curriculum] CEIA checkpoint not found — staying with frozen-self")
                self._stage = 2
                self._iters_in_stage = 0
                return
            opp = _load_ceia_opponent()
            label = "CEIA"

        else:
            return

        _update_all_workers(trainer, opp)
        print(f"[Curriculum] Promoted to stage {new_stage} ({label}), "
              f"episode_reward_mean={mean:.3f}")


# ---------------------------------------------------------------------------
# Reward-shaping wrapper
# ---------------------------------------------------------------------------

class RewardShaperWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_ball_x = None

    def reset(self):
        obs = self.env.reset()
        self._prev_ball_x = None
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)
        shaping = 0.0

        if isinstance(info, dict) and "ball_info" in info:
            ball_x = float(info["ball_info"]["position"][0])

            if self._prev_ball_x is not None:
                shaping += COEFF_BALL_TOWARD_GOAL * (ball_x - self._prev_ball_x)
            self._prev_ball_x = ball_x

            norm_x = np.clip(ball_x / FIELD_HALF_X, -1.0, 1.0)
            shaping += COEFF_BALL_ZONE * norm_x

            if "player_info" in info:
                vel = info["player_info"]["velocity"]
                speed = np.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2)
                if speed < 0.1:
                    shaping -= COEFF_STILLNESS

        return obs, base_reward + shaping, done, info


# ---------------------------------------------------------------------------
# Env creator
# ---------------------------------------------------------------------------

def create_env(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    # Start at stage 0: still opponent
    env_config = dict(env_config)
    env_config.setdefault("opponent_policy", _still_opponent)
    env = __import__("soccer_twos").make(**env_config)
    return RewardShaperWrapper(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("SoccerCurriculum", create_env)

    analysis = tune.run(
        "PPO",
        name="PPO_curriculum_reward",
        config={
            "num_gpus": 0,
            "num_workers": 16,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "SoccerCurriculum",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
            },
            "callbacks": CurriculumCallback,
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
            "timesteps_total": 40_000_000,
        },
        checkpoint_freq=200,
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
