

# ===>>>
import numpy as np
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt

import torch

# ===>>>
# This is the 7th of all files for WGAN on Sepsis

# ===>>>
ReadFrom = "../data/"
BST_Float = torch.load(ReadFrom + "A001_BTS_Float")
BST_nonFloat = torch.load(ReadFrom + "A001_BTS_nonFloat")


# ===>>>
# Below, we will back-transform our data,
# for both the real data and the synthetic ones,
# to their original states
# using previously saved variables in script A001
def Execute_C007(Data):
    # ---
    # We will start with the float variables
    Reverted_Data = Data[:, :36]

    for itr in range(36):
        CurVal = Reverted_Data[:, itr]

        # which we mainly applied MinMaxScale to
        # hasten training
        CurMinX0 = BST_Float["min_X0"][itr]
        CurMaxX1 = BST_Float["max_X1"][itr]
        CurLN = BST_Float["LogNormal"][itr]

        CurVal = CurVal * CurMaxX1 + CurMinX0

        # Then, additional steps are required for
        # those that originally followed a log normalisastion
        if CurLN:
            CurVal = torch.exp(CurVal) - 1

        # And since our original data was extracted based on Komorowski's AI
        # Clinician we realised that their original Age variable was counted
        # in days instead of years slight readjustments were hence made base
        # on Line 786 of AIClinician_sepsis3_def_160219.m in
        # https://gitlab.doc.ic.ac.uk/AIClinician/AIClinician/-/blob/master
        # /AIClinician_sepsis3_def_160219.m
        if itr == 0:
            CurVal = CurVal / 365.25

        Reverted_Data[:, itr] = CurVal

    # ---
    # Afterwards, we move on to the binary variables
    Demo001 = Data[:, 36:38]
    Demo003 = Data[:, 38:40]
    Vent001 = Data[:, 40:42]

    # of which we find the most suitable groups
    _, Demo001 = Demo001.topk(1, dim=1)
    _, Demo003 = Demo003.topk(1, dim=1)
    _, Vent001 = Vent001.topk(1, dim=1)

    Demo001 = Demo001.float()
    Demo003 = Demo003.float()
    Vent001 = Vent001.float()

    # ---
    # And similarly for the categorical ones

    Vitl001 = Data[:, 42:45]
    Vitl002 = Data[:, 45:58]
    Vitl007 = Data[:, 58:68]
    Labs018 = Data[:, 68:78]
    Labs019 = Data[:, 78:88]
    Labs020 = Data[:, 88:98]

    # but remember that GCS was rescaled to 0 - 12
    # for the purpose of embedding,
    # but its raw values only made when
    # it is in a scale of 3 - 15
    _, Vitl002 = Vitl002.topk(1, dim=1)
    Vitl002 = Vitl002 + 3
    Vitl002 = Vitl002.float()

    # for those CAT_NLN variables
    # we will continue to treat them as categorical variables
    _, Vitl001 = Vitl001.topk(1, dim=1)
    Vitl001 = Vitl001.float()

    _, Vitl007 = Vitl007.topk(1, dim=1)
    Vitl007 = Vitl007.float()

    _, Labs018 = Labs018.topk(1, dim=1)
    Labs018 = Labs018.float()

    _, Labs019 = Labs019.topk(1, dim=1)
    Labs019 = Labs019.float()

    _, Labs020 = Labs020.topk(1, dim=1)
    Labs020 = Labs020.float()

    # ---
    Reverted_Data = torch.cat(
        (Reverted_Data,
         Demo001,
         Demo003,
         Vent001,
         Vitl001,
         Vitl002,
         Vitl007,
         Labs018,
         Labs019,
         Labs020
         ), dim=1)

    return Reverted_Data
