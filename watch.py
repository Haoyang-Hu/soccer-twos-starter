import logging
import os
import subprocess
import gym
import numpy as np
import soccer_twos
from soccer_twos.utils import get_agent_class
import importlib
import argparse


class FakeEnv:
    """Lightweight stand-in that provides observation/action spaces without launching Unity."""
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3, 3])


if __name__ == "__main__":
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    parser = argparse.ArgumentParser(description="Rollout soccer-twos.")
    parser.add_argument("-m1", "--agent1-module", required=True, help="Team 1 agent module")
    parser.add_argument("-m2", "--agent2-module", required=True, help="Team 2 agent module")
    parser.add_argument("-p", "--base-port", type=int, help="Base communication port")
    args = parser.parse_args()

    # Kill any leftover Unity soccer-twos processes from previous runs
    subprocess.run("pkill -f soccer-twos", shell=True, stderr=subprocess.DEVNULL)

    # Initialize agents with a fake env (no Unity launched)
    logging.info("Initializing agents...")
    agent1_module = importlib.import_module(args.agent1_module)
    agent2_module = importlib.import_module(args.agent2_module)
    fake_env = FakeEnv()
    agent1 = get_agent_class(agent1_module)(fake_env)
    agent2 = get_agent_class(agent2_module)(fake_env)
    logging.info("Agents initialized. Starting watch env...")

    # Launch fresh Unity env — agents are ready, reset fires immediately
    env = soccer_twos.make(
        watch=True,
        base_port=args.base_port,
        blue_team_name=agent1.name,
        orange_team_name=agent2.name,
    )
    obs = env.reset()

    # Game loop with Python-side score tracking
    blue_score = 0
    orange_score = 0
    episode = 0
    while True:
        agent1_actions = agent1.act({0: obs[0], 1: obs[1]})
        agent2_actions = agent2.act({0: obs[2], 1: obs[3]})
        actions = {
            0: agent1_actions[0],
            1: agent1_actions[1],
            2: agent2_actions[0],
            3: agent2_actions[1],
        }
        obs, reward, done, info = env.step(actions)

        if max(done.values()):
            episode += 1
            ep_reward_blue = reward[0] + reward[1]
            ep_reward_orange = reward[2] + reward[3]
            if ep_reward_blue > ep_reward_orange:
                blue_score += 1
            elif ep_reward_orange > ep_reward_blue:
                orange_score += 1
            logging.info(
                f"Episode {episode} | "
                f"Score: {agent1.name} {blue_score} - {orange_score} {agent2.name}"
            )
            obs = env.reset()
