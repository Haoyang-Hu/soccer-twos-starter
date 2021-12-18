import multiprocessing
import logging
import os

import json_tricks
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from tqdm import tqdm

from transition_recorder_wrapper import TransitionRecorderWrapper


RAW_DS_PATH = "./data/raw"
PROCESSED_DS_PATH = "./data/processed"


def process_file(file_path):
    if not file_path.endswith(".jsonl"):
        return

    with open(os.path.join(RAW_DS_PATH, file_path), "r") as f:
        # each line on this file is an episode
        episode_s = f.readline()
        while episode_s:
            episode = json_tricks.loads(episode_s, ignore_comments=False)
            if type(episode) is str:
                episode = TransitionRecorderWrapper.uncompress_data(episode)

            for t, obs, action, new_obs, reward, done in zip(
                range(len(episode["obs"])),
                episode["obs"],
                episode["action"],
                episode["new_obs"],
                episode["reward"],
                episode["done"],
            ):
                # any MDP-related reprocessing would happen here

                # add to batch
                batch_builder.add_values(
                    t=t,
                    eps_id=episode["episode_id"],
                    agent_index=episode["agent_id"],
                    obs=obs,
                    actions=action,
                    action_prob=1.0,  # put the true action probability here
                    action_logp=0.0,
                    rewards=reward,
                    dones=done,
                    new_obs=new_obs,
                )

            # write file & load new episode
            writer.write(batch_builder.build_and_reset())
            episode_s = f.readline()


if __name__ == "__main__":
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(
        PROCESSED_DS_PATH,
        compress_columns=[
            "t",
            "eps_id",
            "agent_index",
            "obs",
            "actions",
            "action_prob",
            "action_logp",
            "rewards",
            "dones",
            "new_obs",
        ],
    )

    logging.info("Starting to process files")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(process_file, os.listdir(RAW_DS_PATH))
    pool.close()
    pool.join()
    logging.info("Finished processing files")
