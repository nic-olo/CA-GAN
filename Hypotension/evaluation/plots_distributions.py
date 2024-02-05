from matplotlib.patches import Patch
import sys
from pdb import set_trace as bp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig


method = "WGAN-GP"  # choose: "SMOTE", "WGAN-GP", "CA-GAN"


def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


plt.rcParams.update({"font.size": 35})

print("Load...")
df_real = pd.read_csv("data/data_real.csv")

if method == "SMOTE":
    df_fake_reference = pd.read_csv("data/GAN_data_fake.csv")
    df_fake = pd.read_csv("data/SMOTE_data.csv")
    df_fake = df_fake[df_fake['icustay_id'].isna()]
    df_fake = df_fake.drop(
        columns=['id', 'icustay_id', 'hour', 'ethnicity_x', 'ethnicity_y'])
    df_fake = df_fake[df_fake_reference.columns]


if method == "WGAN-GP":
    df_fake = pd.read_csv("data/GAN_data_fake.csv")


if method == "CA-GAN":
    df_fake = pd.read_csv("data/CGAN_data_fake.csv")


df_real = df_real[df_fake.columns]

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
df_fake.rename(columns=mapper, inplace=True)
df_real["Type"] = "Real"
df_fake["Type"] = "Synthetic"
df_all = pd.concat([df_real, df_fake], ignore_index=True)
all_cols = mapper.values()

mapper_fluid_boluses = {"0": "<250",
                        "250": "<500", "500": "<1000", "1000": "≥1000"}
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
#### ### Distributions  ### ###
###############################

print("Plot distributions...")

ncols = 5
nrows = 4
figsize = (50, 40)


fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
color = ["r", "deepskyblue"]
sns.set_palette(color)
sns.set_style("ticks")
sns.despine()
for ix, col in enumerate(all_cols):
    i, j = ix // ncols, ix - (ix // ncols) * ncols
    if ix < 9:
        print(col)
        plot = sns.kdeplot(
            x=df_all[col].astype(float).values,
            hue=df_all["Type"],
            fill=True,
            linewidth=3,
            common_norm=False,
            ax=ax[i, j],
            alpha=0.4
        )

        if col == "MAP [mmHg]":
            ax[i, j].set(xlim=[10, 140])
            ax[i, j].set_title('MAP', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mmHg', fontsize=35)
        elif col == "Diastolic BP [mmHg]":
            ax[i, j].set(xlim=[10, 120])
            ax[i, j].set_title('Diastolic BP', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mmHg', fontsize=35)
        elif col == "Systolic BP [mmHg]":
            ax[i, j].set(xlim=[30, 200])
            ax[i, j].set_title('Systolic BP', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mmHg', fontsize=35)
        elif col == "Urine [mL]":
            ax[i, j].set(xlim=[0, 600])
            ax[i, j].set_title('Urine', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mL', fontsize=35)
        elif col == "ALT [IU/L]":
            ax[i, j].set(xlim=[0, 300])
            ax[i, j].set_title('ALT', fontsize=60)
            ax[i, j].set_xlabel(xlabel='IU/L', fontsize=35)
        elif col == "AST [IU/L]":
            ax[i, j].set(xlim=[0, 300])
            ax[i, j].set_title('AST', fontsize=60)
            ax[i, j].set_xlabel(xlabel='IU/L', fontsize=35)
        elif col == "PO2 [mmHg]":
            ax[i, j].set(xlim=[30, 200])
            ax[i, j].set_title('PO2', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mmHg', fontsize=35)
        elif col == "Lactic Acid [mmol/L]":
            ax[i, j].set(xlim=[0, 5])
            ax[i, j].set_title('Lactic Acid', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mmol/L', fontsize=35)
        elif col == "Serum Creatinine [mg/dL]":
            ax[i, j].set(xlim=[0, 8])
            ax[i, j].set_title('Serum Creatinine', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mg/dL', fontsize=35)
    else:
        print(col)
        plot = sns.histplot(
            x=df_all[col],
            hue=df_all["Type"],
            fill=True,
            multiple="dodge",
            stat="probability",
            common_norm=False,
            shrink=0.8,
            alpha=0.5,
            linewidth=0,
            ax=ax[i, j]
        )
        if col == "Urine (M)":
            ax[i, j].set_title('Urine (M)', fontsize=60)
            ax[i, j].set(xlabel=None)
        elif col == "ALT/AST (M)":
            ax[i, j].set_title('ALT/AST (M)', fontsize=60)
            ax[i, j].set(xlabel=None)
        elif col == "FiO2 (M)":
            ax[i, j].set_title('FiO2 (M)', fontsize=60)
            ax[i, j].set(xlabel=None)
        elif col == "GCS (M)":
            ax[i, j].set_title('GCS (M)', fontsize=60)
            ax[i, j].set(xlabel=None)
        elif col == "PO2 (M)":
            ax[i, j].set_title('PO2 (M)', fontsize=60)
            ax[i, j].set(xlabel=None)
        elif col == "Lactic Acid (M)":
            ax[i, j].set_title('Lactic Acid (M)', fontsize=60)
            ax[i, j].set(xlabel=None)
        elif col == "Serum Creatinine (M)":
            ax[i, j].set_title('Serum Creatinine (M)', fontsize=60)
            ax[i, j].set(xlabel=None)
        elif col == "Fluid Boluses [mL]":
            ax[i, j].set_title('Fluid Boluses', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mL', fontsize=35)
        elif col == "Vasopressors [mcg/kg/min]":
            ax[i, j].set_title('Vasopressors', fontsize=60)
            ax[i, j].set_xlabel(xlabel='mcg/kg/min', fontsize=35)
        elif col == "FiO2":
            ax[i, j].set_title('FiO2', fontsize=60)
            ax[i, j].set_xlabel(xlabel='fraction', fontsize=35)
        elif col == "GCS (Total)":
            ax[i, j].set_title('GCS', fontsize=60)
            ax[i, j].set_xlabel(xlabel='point', fontsize=35)

    plot.legend_.remove()
    ax[i, j].yaxis.set_visible(False)
    # ax[i, j].set_xlabel(xlabel=col)
    ax[i, j].tick_params(labelsize=30)

fig.tight_layout(h_pad=3, w_pad=1)

legend_elements = [Patch(facecolor='r', edgecolor='r', label='Real Data', alpha=0.5, linewidth=3),
                   Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='%s Synthetic Data' % method, alpha=0.5, linewidth=3)]
fig.legend(handles=legend_elements, loc="lower center",
           borderaxespad=0.1, ncol=2, fontsize=45)

# plt.subplots_adjust(right=0.85)
plt.subplots_adjust(bottom=0.07)

fig.savefig("plot/distributions/%s_distributions.png" % method)
