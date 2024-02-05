import numpy as np
from scipy import ndimage
import pandas as pd
import statistics

EPS = 0.000000000000000000001


def kl(x, y, sigma=1):
    bins = (10, 10)
    # histogram
    hist_xy = np.histogram2d(x, y, bins=bins)[0]

    # smooth it out for better results
    ndimage.gaussian_filter(hist_xy, sigma=sigma,
                            mode='constant', output=hist_xy)

    # compute marginals
    hist_xy = hist_xy + EPS  # prevent division with 0
    hist_xy = hist_xy / np.sum(hist_xy)
    hist_x = np.sum(hist_xy, axis=0)
    hist_y = np.sum(hist_xy, axis=1)

    kl = -np.sum(hist_x * np.log(hist_y / hist_x))
    return kl


# Reading the data
data_black_real = pd.read_csv("data/data_real.csv", sep=',')
data_black_fake_gan = pd.read_csv("data/GAN_data_fake.csv", sep=',')
data_black_fake_cgan = pd.read_csv(
    "data/CGAN_data_fake.csv", sep=',')
data_fake_smote = pd.read_csv("data/SMOTE_data.csv", sep=',')

data_black_fake_smote = data_fake_smote[data_fake_smote['icustay_id'].isna()]


# drop indices and reorder columns
data_black_real = data_black_real.drop(columns=['index', 'icustay_id', 'hour'])
data_black_fake_smote = data_black_fake_smote.drop(
    columns=['id', 'icustay_id', 'hour', 'ethnicity_x', 'ethnicity_y'])

data_black_fake_gan = data_black_fake_gan[data_black_real.columns]
data_black_fake_cgan = data_black_fake_cgan[data_black_real.columns]
data_black_fake_smote = data_black_fake_smote[data_black_real.columns]

# if data_fake has more rows than data_real it removes them
# data_fake_cond = data_fake_cond.head(data_real_cond.shape[0])
# data_real_cond = data_real_cond.head(data_fake_cond.shape[0])

# data_real_smote = data_real.head(data_smote.shape[0])

# data chunks
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

# divergence for each column

for col in data_black_real.columns:
    print(col, ":")

    temp_gan_results_list = []
    temp_cgan_results_list = []
    temp_smote_results_list = []

    for i in range(0, len(list_gan)):
        print(i)
        if (data_black_real.shape[0] == list_gan[i].shape[0]):

            # list of distribution
            data_real_feature = np.squeeze(data_black_real[[col]]).tolist()
            data_gan_feature = np.squeeze(list_gan[i][[col]]).tolist()
            data_cgan_feature = np.squeeze(list_cgan[i][[col]]).tolist()
            data_smote_feature = np.squeeze(list_smote[i][[col]]).tolist()

            # kl divergence
            kl_div_gan = kl(data_real_feature, data_gan_feature)
            kl_div_cgan = kl(data_real_feature, data_cgan_feature)
            kl_div_smote = kl(data_real_feature, data_smote_feature)

            print(" original: ", kl_div_gan)
            print(" conditional: ", kl_div_cgan)
            print(" smote: ", kl_div_smote)

            temp_gan_results_list.append(kl_div_gan)
            temp_cgan_results_list.append(kl_div_cgan)
            temp_smote_results_list.append(kl_div_smote)

    # save partial results
    gan_results_list.append(statistics.mean(temp_gan_results_list))
    cgan_results_list.append(statistics.mean(temp_cgan_results_list))
    smote_results_list.append(statistics.mean(temp_smote_results_list))
    col_list.append(col)

# save csv
results_df = pd.DataFrame({"variable": col_list, "GAN": gan_results_list,
                          "CGAN": cgan_results_list, "SMOTE": smote_results_list})
results_df.to_csv("kl_divergence_results.csv", index=False)
