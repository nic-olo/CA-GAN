import pickle
import sys
from pdb import set_trace as bp
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils
from lib.utils import load_obj
from lib.ca-gan import WGAN_GP
from omegaconf import DictConfig
from scipy.special import inv_boxcox
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    artifacts_dir = "mlruns/0/93ba0a9228b0476e9fd3de851c60b6ec/artifacts/"
    print("Load data types...")
    dtype = {
        "name": "str",
        "type": "str",
        "num_classes": "int32",
        "embedding_size": "int32",
        "index_start": "int32",
        "index_end": "int32",
        "include": "bool",
    }
    data_types = pd.read_csv(
        artifacts_dir + "data_types.csv", usecols=dtype.keys(), dtype=dtype
    )
    data_types = data_types[data_types["include"]]

    transforms_dict = load_obj(hp.dir.data + "transforms_dict.pkl")

    print("Generate synthetic data...")
    wgan_gp = WGAN_GP(hp, data_types)
    wgan_gp.load_model(
        artifacts_dir + f"models_generator/48/",
        artifacts_dir + f"models_discriminator/48/",
    )
    labels = [1] * (hp.num_samples)
    labels = torch.from_numpy(pd.DataFrame(labels).values).int()
    labels = torch.Tensor(labels).reshape(-1, 1)
    data_fake = wgan_gp.generate_data(hp.num_samples,
                                      to_np=True,
                                      labels=labels,
                                      )
    data_fake = data_fake.reshape(
        (hp.num_samples * hp.max_sequence_length, -1)
    )

    # Add columns
    df_real = pd.read_pickle(hp.dir.data + "data_real_transformed.pkl")
    df_real = df_real.drop(["icustay_id", "hour", "ethnicity"], axis="columns")
    df_fake = pd.DataFrame(data_fake, columns=df_real.columns)

    print("Save...")
    df_fake.to_pickle(hp.dir.data + "data_fake_transformed.pkl")
    df_fake.to_csv(hp.dir.data + "data_fake_transformed.csv", index=False)

    print("Transform back...")
    df_fake = pd.DataFrame()
    for index, row in data_types.iterrows():
        # inverse real transforms
        if row["type"] == "real":
            minmax_scaler = transforms_dict[row["name"]]["minmax_scaler"]
            lambda_bc = transforms_dict[row["name"]]["lambda_bc"]
            tmp = data_fake[:, row["index_start"]: row["index_end"]]
            tmp = minmax_scaler.inverse_transform(tmp)
            tmp = inv_boxcox(tmp, lambda_bc) - 1
            df_fake[row["name"]] = tmp.squeeze()
        # inverse of categorical dummification
        if row["type"] == "cat" or row["type"] == "bin":
            tmp = pd.DataFrame(
                data_fake[:, row["index_start"]: row["index_end"]],
                columns=transforms_dict[row["name"]]["dummy_columns"],
            )
            df_fake[row["name"]] = tmp.idxmax(axis="columns")

    print("Save...")
    df_fake.to_pickle(hp.dir.data + "data_fake.pkl")
    df_fake.to_csv(hp.dir.data + "data_fake.csv", index=False)


if __name__ == "__main__":
    # Disable logging for this script
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=disabled")
    sys.argv.append("hydra/hydra_logging=disabled")

    main()
