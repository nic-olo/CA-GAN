import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data as utils


###===>>>
def Execute_C003(
        df, data_types, batch_size):
    ###===###
    # There's a couple of few differences between
    # the Hypotension dataset and the Sepsis dataset
    # the most important feature that influences training
    # is that not all patients have an equal amount of data points
    # this ranges from 1 all the way up to 20
    mimictable = pd.read_csv('../data/mimictable15.csv')
    minority = mimictable.drop(mimictable[
                               (mimictable['gender'] != 1.0)].index,
                               inplace=False)
    minority_ids = np.array(minority['icustayid'])
    df = df.loc[(df['Admn001_ID']).isin(minority_ids)]
    df.reset_index(drop=True, inplace=True)
    my_patients = df['Admn001_ID']
    df = df.drop(['Admn001_ID'], axis=1)
    # print(df.shape)
    ###===###
    # So it becomes important for us to grab those data
    # of a specific length and furthermore to be able to
    # find patients of a particular length
    Dict_Len2IDs = {}
    Dict_ID2Rows = {}

    ###===>>>
    # (Part 1)      Find the rows of data of each patient
    for itr in range(my_patients.shape[0]):

        # ---
        Cur_Patient = my_patients[itr]

        # ---
        if Cur_Patient not in Dict_ID2Rows.keys():
            Dict_ID2Rows[Cur_Patient] = [itr]

        else:
            Dict_ID2Rows[Cur_Patient].append(itr)

    ###===>>>
    # (Part 2)      Find those patients of the same length
    for itr in range(len(Dict_ID2Rows)):

        # ---
        Cur_Patient = list(Dict_ID2Rows.keys())[itr]
        Len_Patient = len(Dict_ID2Rows[Cur_Patient])

        # ---
        if Len_Patient not in Dict_Len2IDs.keys():
            Dict_Len2IDs[Len_Patient] = [Cur_Patient]

        else:
            Dict_Len2IDs[Len_Patient].append(Cur_Patient)

    ###===>>>
    # (Part 3)      Preparing the loaders which iterates data of
    #               [BatchSize, Length, Features]
    All_Loader = {}

    # ---
    for itr in range(len(Dict_Len2IDs)):

        # ---
        Cur_Len = list(Dict_Len2IDs.keys())[itr]

        Temp_DF = pd.DataFrame()
        Cur_IDs = Dict_Len2IDs[Cur_Len]

        # ---
        for IndID in Cur_IDs:
            IdRows = Dict_ID2Rows[IndID]

            Temp_DF = Temp_DF.append(df.iloc[IdRows])

        # ---
        data = Temp_DF.values

        # This is the part which we reshape the data into the desired shape
        data = data.reshape(
            (-1, Cur_Len, max(data_types["index_end"]))
        )

        num_patients = data.shape[0]
        data = utils.TensorDataset(
            torch.from_numpy(data).float(),
            torch.full((num_patients, 1, 1), Cur_Len),
        )
        trn_loader = utils.DataLoader(
            data, batch_size=batch_size, shuffle=True, drop_last=True
        )

        All_Loader[Cur_Len] = trn_loader

    ###===>>>
    # (Part 4)      All trainable data
    # Below, we will also need to return a complete list of trainable data
    # This is done in order to calculate the correlations among variables
    all_trainable_data = []

    # ---
    for Cur_Key in All_Loader.keys():
        cur_loader = All_Loader[Cur_Key]

        # ---
        for batch_idx, (x, _) in enumerate(cur_loader):
            all_trainable_data.append(x)

    # ---
    all_trainable_data = torch.cat(all_trainable_data, dim=1)

    ###===###
    return Dict_Len2IDs, Dict_ID2Rows, All_Loader, all_trainable_data


###===>>>
# (Part 0)      Seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

###===>>>
# (Part 1)      Set up the hyperparameters of this script
# ---
# Experimental setup
Hyper001_BatchSize = 32
Hyper002_Epochs = 400

# ---
# Training setup
Hyper003_G_iter = 5  # the ratio of critic-generator update
Hyper004_GP_Lambda = 10  # the regularisation weight of gradient penalty
Hyper005_C_Lambda = 10  # the regularisation weight of correlation alignment

# ---
# Network setup
Hyper006_ID = 128  # the sampling input dimension (ID) and the generator
Hyper007_HD = 128  # the hidden dimension (HD) used across the generator &
# discriminator

# ---
# Optimisation setup
Hyper008_LR = 1e-3  # the learning rate used for both the generator &
# discriminator
Hyper009_Betas = (
    0.9, 0.99)  # the moment coefficients used in the Adam optimiser

# ---
# Continue to train a pre-trained variant
Hyper0010_Continue_YN = True
Hyper0011_G_SD = 'B002_G_StateDict_Epoch300'
Hyper0012_D_SD = 'B002_D_StateDict_Epoch300'
Hyper0013_PreEpoch = 300

###===>>>
# (Part 2)      Load the transformed data, and the reference sheet
df = pd.read_csv("../data/g_A001_data_real_transformed.csv")


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
    "../data/g_A001_data_types.csv",
    usecols=dtype.keys(),
    dtype=dtype,
    index_col="index",
)
data_types = data_types[data_types["include"]]

###===>>>
# (Part 3)      Prepare the dataloaders for training WGAN
Dict_Len2IDs, Dict_ID2Rows, All_Loader, All_Trainable_Data = \
    Execute_C003(df, data_types, Hyper001_BatchSize)

torch.save(All_Trainable_Data, '../data/g_All_Trainable_Data')
