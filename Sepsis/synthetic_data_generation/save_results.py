import sys

# ===>>>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import hydra
from lib.ca-gan import WGAN_GP
import torch
from tqdm import tqdm

# ---
from C007_Utils_BackTransform import *
from C008_Utils_ReplaceNames import *
import argparse
import os


num_generations = 5


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    global num_generations
    original_cwd = hydra.utils.get_original_cwd() + "/"
    data_types = pd.read_csv(
        original_cwd + hp.dir.config + "data_types.csv")

    artifacts_dir = "./mlruns/0/8674f82cd92f40cabd8bfe9d8e0752ae/artifacts/"

    # ===>>>
    # Below, we will compare our real, and fake data
    wgan_gp = WGAN_GP(hp, data_types)
    wgan_gp.load_model(
        artifacts_dir + f"models_generator/{hp.max_sequence_length}/",
        artifacts_dir + f"models_discriminator/{hp.max_sequence_length}/",
    )
    dtype = {
        "index": "int32",
        "name": "str",
        "type": "str",
        "num_classes": "int32",
        "embedding_size": "int32",
        "index_start": "int32",
        "index_end": "int32",
        "include": "bool",
    }
    data_types = pd.read_csv(
        original_cwd + hp.dir.config + "data_types.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        index_col="index",
    )

    read_from = "../data/"
    all_trainable_data = torch.load(read_from + "All_Trainable_Data")

    # ===>>>
    # (Part 1)      Data selection
    # In order to present our data easily,
    # we will randomly select a subset of the dataset for feature comparison
    Real_Data = all_trainable_data.view(-1, 98)
    for gen in tqdm(range(4, num_generations)):
        generation_dir = f"{hp.dir.data}generation_{gen}/"

        if not os.path.exists(generation_dir):
            os.makedirs(generation_dir)

        labels = [1] * hp.num_samples
        labels = torch.from_numpy(pd.DataFrame(labels).values).int()
        labels = labels = labels.clone().detach().reshape(-1, 1)
        temp = wgan_gp.generate_data(hp.num_samples, labels=labels)
        Fake_Data = temp.view(-1, 98)
        print(Fake_Data.shape)
        print("=====================================")

        # ===>>>
        # (Part 2)      Feature back-transformation
        # And before we compare the data, we will need to
        # back-transform them to the state before re-scaled
        Real_Data = Execute_C007(Real_Data)
        real_data = pd.DataFrame(
            Real_Data.cpu().detach().numpy())
        Fake_Data = Execute_C007(Fake_Data)
        fake_data = pd.DataFrame(
            Fake_Data.cpu().detach().numpy())
        fake_data.to_csv(
            original_cwd + generation_dir + "fake_data.csv",
            index=False)
        real_data.to_csv(
            original_cwd + generation_dir + "real_data.csv",
            index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake data.")
    parser.add_argument('--num_generations', type=int,
                        default=5, help="Number of fake data generations.")
    args, unknown = parser.parse_known_args()
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=disabled")
    sys.argv.append("hydra/hydra_logging=disabled")
    num_generations = args.num_generations

    main()
