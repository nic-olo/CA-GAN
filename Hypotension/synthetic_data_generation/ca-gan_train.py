import pickle
from pdb import set_trace as bp

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils
from torch import Tensor
from torch.utils.data import Dataset

from lib.utils import save_obj, get_device
from lib.ca-gan import WGAN_GP, correlation
from omegaconf import DictConfig
from typing import Tuple


class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor, labels) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in
                   tensors), "Size mismatch between tensors"
        assert tensors[0].shape[0] == len(
            labels), "Length mismatch between data and labels"

        self.tensors = tensors
        self.labels = labels

    def __getitem__(self, index):
        temp = tuple(tensor[index] for tensor in self.tensors)
        return temp, self.labels[index]

    def __len__(self):
        return self.tensors[0].size(0)


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    device = get_device(1)
    original_cwd = hydra.utils.get_original_cwd() + "/"

    print("Load...")
    df = pd.read_pickle(
        original_cwd + hp.dir.data + "data_real_transformed.pkl")
    df = df.drop(["icustay_id", "hour"], axis=1)
    ethnicity_labels = df["ethnicity"]
    df = df.drop(["ethnicity"], axis=1)
    print("Load data types...")
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
    data_types = data_types[data_types["include"]]

    print("Create data loaders and tensors...")
    data = df.values
    data = data.reshape(
        (-1, hp.max_sequence_length, max(data_types["index_end"]))
    )  # (3910, 48, 54)
    labels = ethnicity_labels.values[::hp.max_sequence_length]
    labels = labels.reshape(
        (-1, 1)
    )  # (3910, 48, 1)

    labels = torch.from_numpy(labels).int()
    data = torch.from_numpy(data).float()

    # print("data.shape: ", data.shape)
    # print("labels.shape: ", labels.shape)
    # exit(0)
    correlation_real = correlation(data)
    num_patients = data.shape[0]
    data = TensorDataset(data,
                         torch.full(
                             (num_patients, 1, 1), hp.max_sequence_length
                         ),
                         labels=labels)
    trn_loader = utils.DataLoader(
        data, batch_size=hp.batch_size, shuffle=True, drop_last=True
    )

    print("Train model...")
    mlflow.set_tracking_uri(original_cwd + hp.dir.mlruns)
    wgan_gp = WGAN_GP(hp, data_types, correlation_real)

    # artifacts_dir = "../../../mlruns/0/c93934a39c414ab1abf056672917a7b4
    # /artifacts/" wgan_gp.load_model( artifacts_dir +
    # "models_generator/48/", artifacts_dir + "models_discriminator/48/", )
    print("URI setted...")
    with mlflow.start_run():
        mlflow.log_params(hp)
        data_types.to_csv("data_types.csv", index=False)
        mlflow.log_artifact("data_types.csv")
        wgan_gp.train(trn_loader)


if __name__ == "__main__":
    main()
