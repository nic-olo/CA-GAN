import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


plt.rcParams.update({"font.size": 35})

print("Load...")
df_real = pd.read_csv("data/real_data.csv")



#"WGAN-GP":
df_gan = pd.read_csv("data/fake_data_GAN.csv")

#"CA-GAN":
df_cgan = pd.read_csv("data/fake_data_cgan.csv", sep=',')

#"SMOTE":
df_smote = pd.read_csv("data/data_smote.csv", sep=',')


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

data_types = pd.read_csv("data/data_types.csv",
    usecols=dtype.keys(),
    dtype=dtype,
    index_col="index",
)


print("Rename columns...")
df_real.columns = list(data_types["name"])
df_gan.columns = list(data_types["name"])
df_cgan.columns = list(data_types["name"])
df_smote.columns = list(data_types["name"])
df_real["Type"] = "Real"
df_gan["Type"] = "WGAN-GP"
df_cgan["Type"] = "CA-GAN"
df_smote["Type"] = "SMOTE"
df_all = pd.concat([df_real, df_gan, df_cgan, df_smote], ignore_index=True)


###############################
#### ### Correlations   ### ###
###############################

print("Plot correlations...")



print("real...")
real_matrix = (
    df_all[df_all["Type"] == "Real"]
    .drop(columns=["Type"])
    .astype(np.float64)
    .corr('kendall')
)
print("smote...")
smote_matrix = (
    df_all[df_all["Type"] == "SMOTE"]#.sample(frac =0.5)
    .drop(columns=["Type"])
    .astype(np.float64)
    .corr('kendall')
)
print("gan...")
gan_matrix = (
    df_all[df_all["Type"] == "WGAN-GP"]#.sample(frac =0.5)
    .drop(columns=["Type"])
    .astype(np.float64)
    .corr('kendall')
)
print("cgan...")
cgan_matrix = (
    df_all[df_all["Type"] == "CA-GAN"]#.sample(frac =0.5)
    .drop(columns=["Type"])
    .astype(np.float64)
    .corr('kendall')
)
mask = np.triu(np.ones_like(real_matrix, dtype=bool))


#plot2x2
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(80, 60))

with sns.axes_style("white"):
    for i, (matrix, data_type) in enumerate(
        zip([real_matrix, cgan_matrix, gan_matrix, smote_matrix], ["Real", "CA-GAN", "WGAN-GP*", "SMOTE"])
    ):
        sns.heatmap(
            matrix,
            cmap="PuOr",
            mask=mask,
            vmin="-1",
            vmax="1",
            linewidths=0.5,
            #square=True,
            ax=ax[i//2,i%2],
            #annot=True,
            xticklabels=True, yticklabels=True
        )
        ax[i//2,i%2].tick_params(labelsize=25)
        if data_type=="CA-GAN":
            ax[i//2,i%2].set_title(data_type, fontsize= 100, fontweight="bold")
        else:
            ax[i//2,i%2].set_title(data_type, fontsize= 100)


fig.tight_layout(pad=3.0)
fig.savefig("plot/kendall.png")
