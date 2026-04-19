# PPO self-play + defensive reward shaping, warm-started from PPO_selfplay
#
# ============================================================================
# MOTIVATION
# ============================================================================
# Evaluation vs CEIA showed strong offensive play (avg 12.8 goals scored) but
# weak defense (avg 6.8 goals conceded — 95 % win rate, but most games 11–13
# rather than clean wins). The agent outscores opponents by attacking
# relentlessly but doesn't actively defend: it leaves its half exposed and
# lets opponents take too many free shots.
#
# This script continues training from the PPO_selfplay checkpoint
# (checkpoint-7500, 60 M timesteps) and keeps every reward term from
# ppo_selfplay.py UNCHANGED, then adds three new defensive terms that only
# fire when the ball is in our own half. The goal is to turn those 11–13 wins
# into 11–2 wins without hurting offensive behaviour.
#
# ============================================================================
# INHERITED OFFENSIVE SHAPING  (terms 1–7 from ppo_selfplay.py, unchanged)
# ============================================================================
#   1. Ball toward goal        +COEFF_BALL_TOWARD_GOAL × Δball_x
#   2. Ball zone               +COEFF_BALL_ZONE × ball_x / 17
#   3. Chase ball (player 0)   +COEFF_CHASE_BALL × Δdist_player_to_ball
#   4. Stillness penalty       −COEFF_STILLNESS per step when speed < 0.1
#   5. Ball velocity toward goal +COEFF_BALL_VEL_GOAL × max(0, ball_vx)
#   6. Goal-zone amplifier     +COEFF_GOAL_ZONE × max(0, (ball_x − 12) / 5)
#   7. Danger-zone penalty     −COEFF_DANGER_ZONE × max(0, (−ball_x − 10) / 7)
#
# ============================================================================
# NEW DEFENSIVE SHAPING  (terms 8–10, only active when ball_x < 0)
# ============================================================================
# The common guard — ball_x < 0 — ensures these terms reward DEFENSIVE skill
# specifically. When the ball is in the attacking half, offensive shaping
# still does the work; these don't interfere.
#
#   8. Goal-side positioning   (+COEFF_GOAL_SIDE when px < ball_x AND ball in own half)
#      Reward the player for being between the ball and our own goal — the
#      classic defender's position. Prevents the agent from over-committing
#      to offense and being caught upfield. Only fires when ball_x < 0.
#
#   9. Clearance bonus         (+COEFF_CLEAR × max(0, ball_vx) when ball_x < 0)
#      Extra reward for the ball moving toward +x (away from our goal) when
#      in own half. Complements term 5 (goal-directed velocity) by adding a
#      second reward layer specifically for clearing the ball under pressure.
#
#  10. Defensive proximity     (+COEFF_DEFENSIVE_PROX × max(0, 1 − dist/10) when ball_x < 0)
#      Reward the player for being close to the ball when it is in own half,
#      motivating active pressuring/tackling rather than waiting for the
#      opponent to shoot. Uses a linear falloff: full bonus at dist=0,
#      zero when dist ≥ 10 m.
#
# Scale: new defensive max ≈ 0.007/step (only in own half) on top of
# existing ~0.015/step offensive max. Total shaping still well below ±2
# sparse goal signal.
#
# ============================================================================
# TRAINING STRATEGY
# ============================================================================
# - Warm-start from PPO_selfplay checkpoint-7500 (the current HU_PPO3 agent).
# - Keep frozen self-play opponent refreshed every FROZEN_UPDATE_INTERVAL.
# - Checkpoint restores timesteps_total to 60 M; run 40 M more → 100 M total.
#
# ============================================================================
# IMPLEMENTATION NOTES
# ============================================================================
# - Everything that was picklable in ppo_selfplay.py stays picklable here.
# - All three new terms guard on ball_x < 0 to avoid accidentally rewarding
#   passive behaviour in the attacking half.

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
    "ray_results/PPO_selfplay/PPO_SoccerSelfPlay_6da8d_00000_0_2026-04-17_20-06-20"
    "/checkpoint_007500/checkpoint-7500",
)

FROZEN_UPDATE_INTERVAL = 200

NUM_ENVS_PER_WORKER = 1
FIELD_HALF_X        = 17.0

# --- inherited offensive coefficients (identical to ppo_selfplay.py) --------
COEFF_BALL_TOWARD_GOAL = 0.01
COEFF_BALL_ZONE        = 0.002
COEFF_CHASE_BALL       = 0.003
COEFF_STILLNESS        = 0.001
COEFF_BALL_VEL_GOAL    = 0.002
COEFF_GOAL_ZONE        = 0.005
COEFF_DANGER_ZONE      = 0.002

# --- new defensive coefficients ---------------------------------------------
COEFF_GOAL_SIDE        = 0.002   # player between ball and own goal (per step)
COEFF_CLEAR            = 0.003   # ball velocity toward +x while in own half
COEFF_DEFENSIVE_PROX   = 0.002   # player close to ball while in own half


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
    """Snapshot the current team policy (672-dim → 18 logits).
    Orange-team opponent receives 336-dim per-player obs → tile to 672,
    then use first 3 branches as the single-player action."""
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
        if self._iters % FROZEN_UPDATE_INTERVAL == 0:
            state_dict = trainer.get_policy("default_policy").get_weights()
            opp = _make_frozen_opponent(state_dict)
            _update_all_workers(trainer, opp)
            mean = result.get("episode_reward_mean", float("nan"))
            print(f"[SelfPlay-Def] Refreshed frozen opponent "
                  f"(iter {self._iters}, mean={mean:.3f})")
        self._iters += 1


# ---------------------------------------------------------------------------
# Reward-shaping wrapper (offensive + defensive)
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

            # --- OFFENSIVE TERMS (inherited, unchanged) ---------------------

            # 1. Ball toward goal (potential-based Δx)
            if self._prev_ball_x is not None:
                shaping += COEFF_BALL_TOWARD_GOAL * (ball_x - self._prev_ball_x)
            self._prev_ball_x = ball_x

            # 2. Ball zone bonus
            shaping += COEFF_BALL_ZONE * np.clip(ball_x / FIELD_HALF_X, -1.0, 1.0)

            # 5. Ball velocity toward opponent goal (shot-power signal)
            ball_vx = 0.0
            if "velocity" in info["ball_info"]:
                ball_vx = float(info["ball_info"]["velocity"][0])
                shaping += COEFF_BALL_VEL_GOAL * max(0.0, ball_vx)

            # 6. Goal-zone amplifier
            shaping += COEFF_GOAL_ZONE * max(0.0, (ball_x - 12.0) / 5.0)

            # 7. Danger-zone penalty
            shaping -= COEFF_DANGER_ZONE * max(0.0, (-ball_x - 10.0) / 7.0)

            # --- NEW DEFENSIVE TERMS (active only when ball is in own half) -
            in_own_half = ball_x < 0.0

            # 9. Clearance bonus: reward +x ball velocity while defending
            if in_own_half:
                shaping += COEFF_CLEAR * max(0.0, ball_vx)

            if "player_info" in info:
                vel = info["player_info"]["velocity"]
                speed = np.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2)
                px = float(info["player_info"]["position"][0])
                pz = float(info["player_info"]["position"][1])
                curr_dist = np.sqrt((px - ball_x) ** 2 + (pz - ball_z) ** 2)

                # 3. Chase ball (potential-based)
                if self._prev_dist_to_ball is not None:
                    shaping += COEFF_CHASE_BALL * (self._prev_dist_to_ball - curr_dist)
                self._prev_dist_to_ball = curr_dist

                # 4. Stillness penalty
                if speed < 0.1:
                    shaping -= COEFF_STILLNESS

                # 8. Goal-side positioning: reward being between ball and own goal
                if in_own_half and px < ball_x:
                    shaping += COEFF_GOAL_SIDE

                # 10. Defensive proximity: reward closing down ball in own half
                if in_own_half:
                    shaping += COEFF_DEFENSIVE_PROX * max(0.0, 1.0 - curr_dist / 10.0)

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
    env_config = dict(env_config)
    env_config.setdefault("opponent_policy", _random_opponent)
    env = __import__("soccer_twos").make(**env_config)
    return RewardShaperWrapper(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("SoccerSelfPlayDef", create_env)

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_defense",
        # Restore from the current best self-play agent (60 M steps, mean ~2.5)
        restore=PRETRAINED_CHECKPOINT,
        config={
            "num_gpus": 0,
            "num_workers": 16,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "SoccerSelfPlayDef",
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
            # Checkpoint restores timesteps_total to ~60M; run 40M more = 100M total
            "timesteps_total": 100_000_000,
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
