import sys
from pdb import set_trace as bp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig



def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


plt.rcParams.update({"font.size": 35})

print("Load...")
df_real = pd.read_csv("data/data_real.csv")



#"WGAN-GP":
df_gan = pd.read_csv("data/GAN_data_fake.csv")

#"CA-GAN":
df_cgan = pd.read_csv("data/CGAN_data_fake.csv")

#"SMOTE":
df_smote = pd.read_csv("data/SMOTE_data.csv")
df_smote = df_smote[df_smote['icustay_id'].isna()]
df_smote = df_smote.drop(columns=['id', 'icustay_id', 'hour','ethnicity_x','ethnicity_y'])
df_smote = df_smote[df_gan.columns]


df_real = df_real[df_gan.columns]

print("Rename columns...")
mapper = {
    "MAP": "MAP [mmHg]",
    "diastolic_bp": "Diastolic BP [mmHg]",
    "systolic_bp": "Systolic BP [mmHg]",
    "urine": "Urine [mL]",
    "ALT": "ALT [IU/L]",
    "AST": "AST [IU/L]",
    "PO2": "PO2 [mmHg]",
    "lactic_acid": "Lactic Acid [mmol/L]",
    "serum_creatinine": "Serum Creatinine [mg/dL]",
    "fluid_boluses": "Fluid Boluses [mL]",
    "vasopressors": "Vasopressors [mcg/kg/min]",
    "FiO2": "FiO2",
    "GCS_total": "GCS (Total)",
    "urine_m": "Urine (M)",
    "ALT_AST_m": "ALT/AST (M)",
    "FiO2_m": "FiO2 (M)",
    "GCS_total_m": "GCS (M)",
    "PO2_m": "PO2 (M)",
    "lactic_acid_m": "Lactic Acid (M)",
    "serum_creatinine_m": "Serum Creatinine (M)",
}
df_real.rename(columns=mapper, inplace=True)
df_gan.rename(columns=mapper, inplace=True)
df_cgan.rename(columns=mapper, inplace=True)
df_smote.rename(columns=mapper, inplace=True)
df_real["Type"] = "Real"
df_gan["Type"] = "WGAN-GP"
df_cgan["Type"] = "CA-GAN"
df_smote["Type"] = "SMOTE"
df_all = pd.concat([df_real, df_gan, df_cgan, df_smote], ignore_index=True)
all_cols = mapper.values()

mapper_fluid_boluses = {"0": "<250", "250": "<500", "500": "<1000", "1000": "≥1000"}
df_all["Fluid Boluses [mL]"] = (
    df_all["Fluid Boluses [mL]"]
    .astype(str)
    .astype("category")
    .map(mapper_fluid_boluses)
)
df_all["Fluid Boluses [mL]"].cat.reorder_categories(
    mapper_fluid_boluses.values(), inplace=True
)

mapper_vasopressors = {
    "0.0": "0",
    "1e-06": "<8.4",
    "8.4": "<20.28",
    "20.28": "≥20.28",
}
df_all["Vasopressors [mcg/kg/min]"] = (
    df_all["Vasopressors [mcg/kg/min]"]
    .astype(str)
    .astype("category")
    .map(mapper_vasopressors)
)
df_all["Vasopressors [mcg/kg/min]"].cat.reorder_categories(
    mapper_vasopressors.values(), inplace=True
)

mapper_FiO2 = {
    "0.0": "<.2",
    "0.2": ".2",
    "0.3": ".3",
    "0.4": ".4",
    "0.5": ".5",
    "0.6": ".6",
    "0.7": ".7",
    "0.8": ".8",
    "0.9": ".9",
    "1.0": "1.0",
}
df_all["FiO2"] = df_all["FiO2"].astype(str).astype("category").map(mapper_FiO2)
df_all["FiO2"].cat.reorder_categories(mapper_FiO2.values(), inplace=True)

mapper_GCS = {
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "10",
    "11": "11",
    "12": "12",
    "13": "13",
    "14": "14",
    "15": "15",
}
df_all["GCS (Total)"] = (
    df_all["GCS (Total)"].astype(str).astype("category").map(mapper_GCS)
)
df_all["GCS (Total)"].cat.reorder_categories(mapper_GCS.values(), inplace=True)

mapper_bin = {"0": "False", "1": "True"}
for col in [
    "Urine (M)",
    "ALT/AST (M)",
    "FiO2 (M)",
    "GCS (M)",
    "PO2 (M)",
    "Lactic Acid (M)",
    "Serum Creatinine (M)",
]:
    df_all[col] = df_all[col].astype(str).astype("category").map(mapper_bin)
    df_all[col].cat.reorder_categories(mapper_bin.values(), inplace=True)

###############################
#### ### Correlations   ### ###
###############################

print("Plot correlations...")

df_all_codes = df_all.copy()
for col in [
    "Fluid Boluses [mL]",
    "Vasopressors [mcg/kg/min]",
    "FiO2",
    "GCS (Total)",
    "Urine (M)",
    "ALT/AST (M)",
    "FiO2 (M)",
    "GCS (M)",
    "PO2 (M)",
    "Lactic Acid (M)",
    "Serum Creatinine (M)",
]:
    df_all_codes[col] = df_all_codes[col].cat.codes

mapper = {
    "MAP [mmHg]": "MAP",
    "Diastolic BP [mmHg]": "Diastolic BP",
    "Systolic BP [mmHg]": "Systolic BP",
    "Urine [mL]": "Urine",
    "ALT [IU/L]": "ALT",
    "AST [IU/L]": "AST",
    "PO2 [mmHg]": "PO2",
    "Lactic Acid [mmol/L]": "Lactic Acid",
    "Serum Creatinine [mg/dL]": "Serum Creatinine",
    "Fluid Boluses [mL]": "Fluid Boluses",
    "Vasopressors [mcg/kg/min]": "Vasopressors",
}
df_all_codes.rename(columns=mapper, inplace=True)

print("real...")
real_matrix = (
    df_all_codes[df_all_codes["Type"] == "Real"]
    .drop(columns=["Type"])
    .astype(np.float64)
    .corr('kendall')
)
print("smote...")
smote_matrix = (
    df_all_codes[df_all_codes["Type"] == "SMOTE"].sample(frac =.50)
    .drop(columns=["Type"])
    .astype(np.float64)
    .corr('kendall')
)
print("gan...")
gan_matrix = (
    df_all_codes[df_all_codes["Type"] == "WGAN-GP"].sample(frac =.50)
    .drop(columns=["Type"])
    .astype(np.float64)
    .corr('kendall')
)
print("cgan...")
cgan_matrix = (
    df_all_codes[df_all_codes["Type"] == "CA-GAN"].sample(frac =.50)
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
        )#.set_title(data_type, fontsize= 100)
        ax[i//2,i%2].set_xticklabels(ax[i//2,i%2].get_xmajorticklabels(), fontsize = 60)
        ax[i//2,i%2].set_yticklabels(ax[i//2,i%2].get_ymajorticklabels(), fontsize = 60)
        if data_type=="CA-GAN":
            ax[i//2,i%2].set_title(data_type, fontsize= 100, fontweight="bold")
        else:
            ax[i//2,i%2].set_title(data_type, fontsize= 100)


fig.tight_layout(pad=3.0)
fig.savefig("plot/kendall.png")
