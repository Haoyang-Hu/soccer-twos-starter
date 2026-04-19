# PPO self-play, warm-started from a pretrained random-opponent checkpoint
#
# ============================================================================
# TRAINING STRATEGY
# ============================================================================
# Restores weights from a PPO checkpoint already trained against a random
# opponent (mean reward ~1.6), then immediately switches to frozen self-play:
#
#   - Every FROZEN_UPDATE_INTERVAL iterations, a numpy snapshot of the current
#     policy is taken and deployed to all workers as the orange-team opponent.
#   - The opponent is always ~200 iterations behind the learner, creating an
#     arms race: the agent must keep improving to stay ahead of its past self.
#   - Opponent difficulty scales automatically — unlike random (fixed, easy
#     long-term) this always challenges at roughly the current skill level.
#
# ============================================================================
# ENHANCED REWARD SHAPING
# ============================================================================
# On top of the sparse ±2 goal signal, seven dense terms guide learning:
#
#   1. Ball toward goal  (+COEFF_BALL_TOWARD_GOAL × Δball_x)
#      Potential-based: rewards every step the ball moves toward +x (opp goal).
#      Does not change the optimal policy, only speeds up credit assignment.
#
#   2. Ball zone         (+COEFF_BALL_ZONE × ball_x / FIELD_HALF_X)
#      Continuous positional bonus for ball in attacking half, penalty in own.
#
#   3. Chase ball        (+COEFF_CHASE_BALL × Δdist_player_to_ball)
#      Potential-based: rewards player 0 for reducing distance to the ball.
#      Motivates active ball-chasing rather than passive waiting.
#      Φ(s) = −dist(player, ball) → F = Φ(s') − Φ(s) = prev_dist − curr_dist
#
#   4. Stillness penalty (−COEFF_STILLNESS per step when speed < 0.1 m/s)
#      Discourages idling; counters the locally-optimal stand-still strategy.
#
#   5. Ball velocity toward goal (+COEFF_BALL_VEL_GOAL × max(0, ball_vx))
#      Uses ball_info["velocity"][0] to directly reward kicking the ball hard
#      toward +x. Complements the potential-based Δx term: that term rewards
#      net displacement; this rewards instantaneous shot power.
#
#   6. Goal-zone amplifier (+COEFF_GOAL_ZONE × max(0, (ball_x − 12) / 5))
#      Non-linear bonus when ball is in the final 5 m before the opponent
#      goal (x > 12). Differentiates merely reaching midfield from actually
#      threatening to score; creates a reward gradient that pulls the ball
#      all the way to the goal line.
#
#   7. Danger-zone penalty (−COEFF_DANGER_ZONE × max(0, (−ball_x − 10) / 7))
#      Mirror of term 6 for own goal: extra penalty when ball is deep in own
#      half (x < −10). Teaches the agent to clear the ball and prevent the
#      frozen-self opponent from threatening.
#
#   Scale: shaping max ~0.015/step vs ±2 for a goal — dense enough to guide
#   early exploration without dominating the true scoring objective.
#
# ============================================================================
# COORDINATE SYSTEM
# ============================================================================
# Soccer-Twos ground plane: position = [x, z], velocity = [vx, vz]
# Blue attacks toward +x, orange toward −x; field x ∈ [−17, 17], z ∈ [−7, 7]
#
# ============================================================================
# IMPLEMENTATION NOTES
# ============================================================================
# - tune.run(restore=...) loads policy weights from the checkpoint; the new
#   env and callback config takes effect immediately.
# - timesteps_total is restored from the checkpoint (~20M), so the stop
#   condition is set to 20M + 40M = 60M total timesteps.
# - worker.foreach_env() handles _VectorEnvToBaseEnv wrapping (NUM_ENVS=1).
# - Frozen snapshot is taken on the very first on_train_result call so workers
#   immediately play against a snapshot of the pretrained policy.

import os
import numpy as np
import gym
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRETRAINED_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results/PPO_shaped/PPO_SoccerShaped_7816f_00000_0_2026-04-17_01-51-25"
    "/checkpoint_002500/checkpoint-2500",
)

FROZEN_UPDATE_INTERVAL = 200   # refresh frozen-self snapshot every N iters

NUM_ENVS_PER_WORKER = 1
FIELD_HALF_X        = 17.0

COEFF_BALL_TOWARD_GOAL = 0.01    # ball moving toward opponent goal (potential Δx)
COEFF_BALL_ZONE        = 0.002   # ball in attacking half bonus
COEFF_CHASE_BALL       = 0.003   # player moving toward ball (potential)
COEFF_STILLNESS        = 0.001   # penalty for standing still
COEFF_BALL_VEL_GOAL    = 0.002   # ball velocity toward opponent goal (+x component)
COEFF_GOAL_ZONE        = 0.005   # extra bonus in final 5 m before opponent goal
COEFF_DANGER_ZONE      = 0.002   # extra penalty when ball is deep in own half


# ---------------------------------------------------------------------------
# Pure-numpy MLP — picklable, no Ray/torch at inference time
# ---------------------------------------------------------------------------

class NumpyMLP:
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
# Frozen opponent factory
# ---------------------------------------------------------------------------

def _make_frozen_opponent(state_dict):
    """
    Snapshot the current team policy (672-dim → 18 logits).
    Orange-team opponent receives 336-dim per-player obs → tile to 672,
    then use first 3 branches as the single-player action.
    """
    mlp = NumpyMLP(state_dict, output_branches=[3, 3, 3, 3, 3, 3])

    def opponent(obs_336):
        team_obs = np.tile(obs_336.astype(np.float32), 2)
        return mlp.act(team_obs)[:3]

    return opponent


def _random_opponent(obs):
    return np.array([np.random.randint(3) for _ in range(3)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Worker update
# ---------------------------------------------------------------------------

def _apply_opponent_to_worker(worker, opponent_fn):
    def _set(env):
        e = env
        while e is not None:
            if hasattr(e, "set_opponent_policy"):
                e.set_opponent_policy(opponent_fn)
                break
            e = getattr(e, "env", None)
    worker.foreach_env(_set)


def _update_all_workers(trainer, opponent_fn):
    for w in trainer.workers.remote_workers():
        w.apply.remote(_apply_opponent_to_worker, opponent_fn)


# ---------------------------------------------------------------------------
# Self-play callback
# ---------------------------------------------------------------------------

class SelfPlayCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self._iters = 0

    def on_train_result(self, trainer, result, **kwargs):
        # Snapshot on iter 0 (immediately) and every FROZEN_UPDATE_INTERVAL after
        if self._iters % FROZEN_UPDATE_INTERVAL == 0:
            state_dict = trainer.get_policy("default_policy").get_weights()
            opp = _make_frozen_opponent(state_dict)
            _update_all_workers(trainer, opp)
            mean = result.get("episode_reward_mean", float("nan"))
            print(f"[SelfPlay] Refreshed frozen opponent "
                  f"(iter {self._iters}, mean={mean:.3f})")
        self._iters += 1


# ---------------------------------------------------------------------------
# Reward-shaping wrapper
# ---------------------------------------------------------------------------

class RewardShaperWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_ball_x       = None
        self._prev_dist_to_ball = None

    def reset(self):
        obs = self.env.reset()
        self._prev_ball_x       = None
        self._prev_dist_to_ball = None
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)
        shaping = 0.0

        if isinstance(info, dict) and "ball_info" in info:
            ball_x = float(info["ball_info"]["position"][0])
            ball_z = float(info["ball_info"]["position"][1])

            # 1. Ball toward goal (potential-based Δx)
            if self._prev_ball_x is not None:
                shaping += COEFF_BALL_TOWARD_GOAL * (ball_x - self._prev_ball_x)
            self._prev_ball_x = ball_x

            # 2. Ball zone bonus
            shaping += COEFF_BALL_ZONE * np.clip(ball_x / FIELD_HALF_X, -1.0, 1.0)

            # 5. Ball velocity toward goal (instantaneous shot-power signal)
            if "velocity" in info["ball_info"]:
                ball_vx = float(info["ball_info"]["velocity"][0])
                shaping += COEFF_BALL_VEL_GOAL * max(0.0, ball_vx)

            # 6. Goal-zone amplifier: extra bonus in final 5 m before opponent goal
            shaping += COEFF_GOAL_ZONE * max(0.0, (ball_x - 12.0) / 5.0)

            # 7. Danger-zone penalty: extra penalty when ball is deep in own half
            shaping -= COEFF_DANGER_ZONE * max(0.0, (-ball_x - 10.0) / 7.0)

            if "player_info" in info:
                vel = info["player_info"]["velocity"]
                speed = np.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2)

                # 3. Chase ball (potential-based: reward reducing dist to ball)
                px = float(info["player_info"]["position"][0])
                pz = float(info["player_info"]["position"][1])
                curr_dist = np.sqrt((px - ball_x) ** 2 + (pz - ball_z) ** 2)
                if self._prev_dist_to_ball is not None:
                    shaping += COEFF_CHASE_BALL * (self._prev_dist_to_ball - curr_dist)
                self._prev_dist_to_ball = curr_dist

                # 4. Stillness penalty
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
    # Default to random until SelfPlayCallback deploys the first snapshot
    env_config = dict(env_config)
    env_config.setdefault("opponent_policy", _random_opponent)
    env = __import__("soccer_twos").make(**env_config)
    return RewardShaperWrapper(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("SoccerSelfPlay", create_env)

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay",
        # Restore pretrained weights (trained ~20M steps vs random, mean ~1.6)
        restore=PRETRAINED_CHECKPOINT,
        config={
            "num_gpus": 0,
            "num_workers": 16,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "SoccerSelfPlay",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
            },
            "callbacks": SelfPlayCallback,
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
            # checkpoint restores timesteps_total to ~20M; run 40M more = 60M total
            "timesteps_total": 60_000_000,
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
