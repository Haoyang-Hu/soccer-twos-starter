import os

import json_tricks
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from tqdm import tqdm

from transition_recorder_wrapper import TransitionRecorderWrapper


RAW_DS_PATH = "./data/raw"
PROCESSED_DS_PATH = "./data/processed"


if __name__ == "__main__":
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(PROCESSED_DS_PATH)

    # load each raw episode group file
    with tqdm(total=0) as pbar:
        for file in os.listdir(RAW_DS_PATH):
            if file.endswith(".jsonl"):
                with open(os.path.join(RAW_DS_PATH, file), "r") as f:
                    # each line on this file is an episode
                    episode_s = f.readline()
                    while episode_s:
                        episode = json_tricks.loads(episode_s, ignore_comments=False)
                        if type(episode) is str:
                            episode = TransitionRecorderWrapper.uncompress_data(episode)

                        pbar.total += len(episode["obs"])
                        pbar.refresh()
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
                            pbar.update(1)

                        # write file & load new episode
                        writer.write(batch_builder.build_and_reset())
                        episode_s = f.readline()
