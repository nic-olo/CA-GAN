import math
import pickle as pkl
from pdb import set_trace as bp

import numpy as np
import pandas as pd
import torch


def save_obj(obj, name):
    with open(name, "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pkl.load(f)


def get_device(n_device=0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{n_device}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
