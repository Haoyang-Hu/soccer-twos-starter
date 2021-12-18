import base64
import logging
import pickle
import os
import time
from typing import Union

import jsonlines
import json_tricks
import lz4.frame

import gym
import numpy as np


class TransitionRecorderWrapper(gym.core.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        save_folder: str = "./data/raw",
        min_transitions_per_file: int = 1000,
        compress=True,
    ):
        super(TransitionRecorderWrapper, self).__init__(env)
        self.env = env
        self.save_folder = save_folder
        self.min_transitions_per_file = min_transitions_per_file
        self.compress = compress

        os.makedirs(save_folder, exist_ok=True)
        logging.info(f"Recording transitions to {save_folder}")

        self.last_obs = None

        self.episode_id = None
        self.file_buffer = []
        self.n_transitions = 0

        self.obs_buffer = None
        self.action_buffer = None
        self.new_obs_buffer = None
        self.reward_buffer = None
        self.done_buffer = None
        self.info_buffer = None

    def step(self, action: Union[int, float, list, np.ndarray, dict]):
        # interact
        obs, reward, done, info = self.env.step(action)
        # record
        self.record_transitions(obs, action, reward, done, info)
        return self.preprocess_obs(obs), reward, done, info

    def reset(self):
        self.save_episode()
        self.reset_episode()
        return self.preprocess_obs(self.env.reset())

    def close(self):
        self.save_episode()
        self.env.close()

    def preprocess_obs(self, obs: np.ndarray):
        self.last_obs = obs
        return obs

    def record_transitions(
        self,
        new_obs: Union[np.ndarray, dict],
        action: Union[int, float, list, np.ndarray, dict],
        reward: Union[float, dict],
        done: Union[bool, dict],
        info: Union[dict, dict],
    ):
        if type(new_obs) is dict:
            for agent_id in new_obs:
                self.record_transition(
                    new_obs[agent_id],
                    action[agent_id],
                    reward[agent_id],
                    done[agent_id] if agent_id in done else done["__all__"],
                    info[agent_id],
                    agent_id,
                )
        else:
            self.record_transition(new_obs, action, reward, done, info)

    def record_transition(
        self,
        new_obs: np.ndarray,
        action: Union[int, float, list, np.ndarray],
        reward: float,
        done: bool,
        info: dict,
        agent_id: int = 0,
    ):
        logging.debug(
            f"Recording transitions: {self.last_obs} {action} {new_obs} {reward} {done} {info}"
        )
        if agent_id not in self.obs_buffer:
            self.obs_buffer[agent_id] = []
            self.action_buffer[agent_id] = []
            self.new_obs_buffer[agent_id] = []
            self.reward_buffer[agent_id] = []
            self.done_buffer[agent_id] = []
            self.info_buffer[agent_id] = []

        self.obs_buffer[agent_id].append(self.last_obs)
        self.action_buffer[agent_id].append(action)
        self.new_obs_buffer[agent_id].append(new_obs)
        self.reward_buffer[agent_id].append(reward)
        self.done_buffer[agent_id].append(done)
        self.info_buffer[agent_id].append(info)
        # no need to record the timestep since it's redundant with relative transition order

    def save_episode(self):
        if not self.obs_buffer or len(self.obs_buffer) == 0:
            return

        any_agent_id = list(self.obs_buffer.keys())[0]
        self.n_transitions += len(self.obs_buffer[any_agent_id])

        logging.info(f"Buffering episode {self.episode_id}")
        logging.info(f"Current buffer size: {self.n_transitions}")

        for agent_id in self.obs_buffer:
            episode = {
                "episode_id": self.episode_id,
                "agent_id": agent_id,
                "obs": self.obs_buffer[agent_id],
                "action": self.action_buffer[agent_id],
                "new_obs": self.new_obs_buffer[agent_id],
                "reward": self.reward_buffer[agent_id],
                "done": self.done_buffer[agent_id],
                "info": self.info_buffer[agent_id],
            }

            if self.compress:
                episode = TransitionRecorderWrapper.compress_data(episode)

            self.file_buffer.append(episode)

        if self.n_transitions >= self.min_transitions_per_file:
            file_path = os.path.join(self.save_folder, f"{self.episode_id}.jsonl")
            logging.info(f"Saving file to {file_path}")
            with open(file_path, "w") as fp:
                writer = jsonlines.Writer(fp, compact=True, dumps=json_tricks.dumps)
                writer.write_all(self.file_buffer)
            self.file_buffer = []
            self.n_transitions = 0

    def reset_episode(self):
        self.episode_id = round(time.time() * 1000)
        self.obs_buffer = {}
        self.action_buffer = {}
        self.new_obs_buffer = {}
        self.reward_buffer = {}
        self.done_buffer = {}
        self.info_buffer = {}

    @staticmethod
    def compress_data(data):
        data = pickle.dumps(data)
        data = lz4.frame.compress(data)
        data = base64.b64encode(data).decode("ascii")
        return data

    @staticmethod
    def uncompress_data(data):
        data = base64.b64decode(data)
        data = lz4.frame.decompress(data)
        data = pickle.loads(data)
        return data
