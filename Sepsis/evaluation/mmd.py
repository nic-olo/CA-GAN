import torch
import pandas as pd

import numpy as np
from sklearn import metrics
import statistics
device = torch.device(  # "cuda" if torch.cuda.is_available() else
    "cpu")


# maximum mean discrepancy function: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
def MMD(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


# load data


data_black_real = pd.read_csv("data/real_data.csv", sep=',')
data_black_fake_gan = pd.read_csv(
    "data/fake_data_GAN.csv", sep=',')
data_black_fake_cgan = pd.read_csv("data/fake_data_cgan.csv", sep=',')
data_black_fake_smote = pd.read_csv(
    "data/data_smote.csv", sep=',')


data_black_fake_cgan = data_black_fake_cgan.head(
    data_black_fake_smote.shape[0])
data_black_fake_gan = data_black_fake_gan.head(data_black_fake_smote.shape[0])

# compute MMD

n = data_black_real.shape[0]  # chunk row size
list_gan = [data_black_fake_gan[i:i+n]
            for i in range(0, data_black_fake_gan.shape[0], n)]
list_cgan = [data_black_fake_cgan[i:i+n]
             for i in range(0, data_black_fake_cgan.shape[0], n)]
list_smote = [data_black_fake_smote[i:i+n]
              for i in range(0, data_black_fake_smote.shape[0], n)]

# lists to save results
col_list = []
gan_results_list = []
cgan_results_list = []
smote_results_list = []


for col in data_black_real.columns:
    print(col, ":")

    temp_gan_results_list = []
    temp_cgan_results_list = []
    temp_smote_results_list = []

    for i in range(0, len(list_gan)):
        print(i)
        if (torch.tensor(data_black_real.values).shape[0] == torch.tensor(list_gan[i].values).shape[0]):
            result = MMD(torch.tensor(data_black_real[col].values).reshape(
                -1, 1), torch.tensor(list_gan[i][col].values).reshape(-1, 1))
            print(f"MMD result of X and Y is {result.item()}")
            temp_gan_results_list.append(result.item())

    for i in range(0, len(list_cgan)):
        print(i)
        if (torch.tensor(data_black_real.values).shape[0] == torch.tensor(list_cgan[i].values).shape[0]):
            result = MMD(torch.tensor(data_black_real[col].values).reshape(
                -1, 1), torch.tensor(list_cgan[i][col].values).reshape(-1, 1))
            print(f"MMD result of X and Y is {result.item()}")
            temp_cgan_results_list.append(result.item())

    for i in range(0, len(list_smote)):
        print(i)
        if (torch.tensor(data_black_real.values).shape[0] == torch.tensor(list_smote[i].values).shape[0]):
            result = MMD(torch.tensor(data_black_real[col].values).reshape(
                -1, 1), torch.tensor(list_smote[i][col].values).reshape(-1, 1))
            print(f"MMD result of X and Y is {result.item()}")
            temp_smote_results_list.append(result.item())

    # save partial results
    gan_results_list.append(statistics.mean(temp_gan_results_list))
    cgan_results_list.append(statistics.mean(temp_cgan_results_list))
    smote_results_list.append(statistics.mean(temp_smote_results_list))
    col_list.append(col)


# save csv
results_df = pd.DataFrame({"variable": col_list,
                           "GAN": gan_results_list,
                           "CGAN": cgan_results_list,
                           "SMOTE": smote_results_list})

results_df.to_csv("mmd_results.csv", index=False)

print("median: ", results_df.median())
print("mean: ", results_df.mean())
