import hydra
import mlflow
import pandas as pd
import torch
import torch.utils.data as utils
from torch import Tensor
from torch.utils.data import Dataset
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

    def __init__(self, *tensors: Tensor, labels=None) -> None:
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
    original_cwd = hydra.utils.get_original_cwd() + "/"
    print("Load...")
    df = pd.read_pickle(
        original_cwd + hp.dir.data + "g_A001_data_real_transformed.pkl")

    df = df.drop(["Admn001_ID"], axis=1)
    gender_labels = pd.read_csv(
        original_cwd + hp.dir.data + "g_A001_labels.csv")['gender']
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
        original_cwd + hp.dir.data + "g_A001_data_types.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        index_col="index",
    )
    data_types = data_types[data_types["include"]]

    print("Create data loaders and tensors...")
    data = df.values
    data = data.reshape(
        (-1, hp.max_sequence_length, max(data_types["index_end"])
         )
    )  # (3910, 48, 54)

    labels = gender_labels.values
    labels = labels.reshape(
        (-1, 1)
    )  # (3910, 48, 1)

    labels = torch.from_numpy(labels).int()
    data = torch.from_numpy(data).float()
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

    print("URI setted...")
    with mlflow.start_run():
        mlflow.log_params(hp)
        data_types.to_csv("data_types.csv", index=False)
        wgan_gp.train(trn_loader)


if __name__ == "__main__":
    main()
