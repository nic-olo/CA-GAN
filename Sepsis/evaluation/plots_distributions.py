from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


method = "SMOTE"  # choose: "SMOTE", "WGAN-GP", "CA-GAN"


def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)


plt.rcParams.update({"font.size": 35})

print("Load...")
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

real_data = pd.read_csv("data/real_data.csv")

if method == "SMOTE":
    fake_data = pd.read_csv("data/data_smote.csv", sep=',')

if method == "WGAN-GP":
    fake_data = pd.read_csv("data/fake_data_GAN.csv")

if method == "CA-GAN":
    fake_data = pd.read_csv("data/fake_data_cgan.csv", sep=',')


# ===>>>
# (Part 3)      Variable comparison
def Execute_C008(data_types):
    Replace_Names = [i for i in list(data_types["name"])]

    Replace_Names[0] = 'Age [days]'
    Replace_Names[1] = 'HR [bpm]'
    Replace_Names[2] = 'SysBP [mmHg]'
    Replace_Names[3] = 'MeanBP [mmHg]'
    Replace_Names[4] = 'DiaBP [mmHg]'

    Replace_Names[5] = 'RR [bpm]'
    Replace_Names[6] = 'K [meq/L]'
    Replace_Names[7] = 'Na [meq/L]'

    Replace_Names[8] = 'Cl [meq/L]'
    Replace_Names[9] = 'Ca [mg/dL]'
    Replace_Names[10] = 'IonisedCa [mg/dL]'
    Replace_Names[11] = 'CO2 [meq/L]'
    Replace_Names[12] = 'Albumin [g/dL]'

    Replace_Names[13] = 'Hb [g/dL]'
    Replace_Names[14] = 'pH'
    Replace_Names[15] = 'BE [meq/L]'
    Replace_Names[16] = 'HCO3 [meq/L]'
    Replace_Names[17] = 'FiO2 [Fraction]'

    Replace_Names[18] = 'Glucose [mg/dL]'
    Replace_Names[19] = 'BUN [mg/dL]'
    Replace_Names[20] = 'Creatinine [mg/dL]'
    Replace_Names[21] = 'Mg [mg/dL]'
    Replace_Names[22] = 'SGOT [u/L]'

    Replace_Names[23] = 'SGPT [u/L]'
    Replace_Names[24] = 'TotalBili [mg/dL]'
    Replace_Names[25] = 'WbcCount [E9/L]'
    Replace_Names[26] = 'PlateletsCount [E9/L]'
    Replace_Names[27] = 'PaO2 [mmHg]'

    Replace_Names[28] = 'PaCO2 [mmHg]'
    Replace_Names[29] = 'Lactate [mmol/L]'
    Replace_Names[30] = 'InputTotal [mL]'
    Replace_Names[31] = 'Input4H [mL]'
    Replace_Names[32] = 'MaxVaso [mcg/kg/min]'

    Replace_Names[33] = 'OutputTotal [mL]'
    Replace_Names[34] = 'Output4H [mL]'
    Replace_Names[35] = 'Gender'
    Replace_Names[36] = 'ReAd'
    Replace_Names[37] = 'Mech'

    Replace_Names[38] = 'Temp'

    Replace_Names[39] = 'GCS'
    Replace_Names[40] = 'SpO2 [%]'
    Replace_Names[41] = 'PTT [s]'
    Replace_Names[42] = 'PT [s]'
    Replace_Names[43] = 'INR'

    return Replace_Names


# We will also do a slight change to the names in data_type to add the
# units
replace_names = Execute_C008(data_types)

BST_nonFloat = torch.load("data/A001_BTS_nonFloat")


###############################
#### ### Distributions  ### ###
###############################

print("Plot distributions...")

""" ncols = 5
nrows = 4
figsize=(50, 40) """

ncols = 5
nrows = 9
figsize = (50, 80)


fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
color = ["r", "deepskyblue"]
sns.set_palette(color)
sns.set_style("ticks")
sns.despine()

for itr in range(44):

    i, j = itr // ncols, itr - (itr // ncols) * ncols

    if itr < 35:

        print(itr)

        cur_name = replace_names[itr]

        cur_fake = fake_data.iloc[:, itr]
        cur_real = real_data.iloc[:, itr]

        # ---
        df_fake = pd.DataFrame()
        df_fake[cur_name] = cur_fake
        df_fake["Type"] = "Synthetic"
        df_real = pd.DataFrame()
        df_real[cur_name] = cur_real
        df_real["Type"] = "Real"
        df_all = pd.concat([df_real, df_fake], ignore_index=True)

        plot = sns.kdeplot(
            x=df_all[cur_name].astype(float).values,
            hue=df_all["Type"],
            fill=True,
            linewidth=3,
            common_norm=False,
            ax=ax[i, j],
            alpha=0.4
        )

        ax[i, j].set_title(cur_name.partition(' [')[0], fontsize=60)
        if '[' in cur_name:
            ax[i, j].set_xlabel(xlabel=cur_name[cur_name.find(
                "[")+1:cur_name.find("]")], fontsize=35)
        else:
            ax[i, j].set_xlabel(xlabel=None)

    else:
        print(itr)

        # ---
        cur_name = replace_names[itr]

        cur_fake = fake_data.iloc[:, itr]
        cur_real = real_data.iloc[:, itr]

        # ---
        df_fake = pd.DataFrame()
        df_fake[cur_name] = cur_fake
        df_fake["Type"] = "Synthetic"
        df_real = pd.DataFrame()
        df_real[cur_name] = cur_real
        df_real["Type"] = "Real"

        # ---
        # below, we set the xticks for the categorical values
        # we first consider the binary variables
        if itr <= 37:

            if itr == 35:
                # for gender, Komorosski has stated that
                # 0 is for male and that 1 is for female
                # in his data description in
                # https://gitlab.doc.ic.ac.uk/AIClinician/AIClinician/-/blob/master/Dataset%20description%20Komorowski%20011118.xlsx
                mapper = {"0": "Male", "1": "Female"}
            else:
                # otherwise, it is just simply False vs True
                mapper = {"0": "False", "1": "True"}

        elif itr == 38:
            # NEW categorical temperature
            mapper = {"0": "<35.05", "1": "35.05-38", "2": ">38"}

        elif itr == 39:
            # and this one here is GCS
            mapper = {str(itr3 + 3): itr3 + 3 for itr3 in range(0, 13)}

        elif itr == 43:
            # INR mapped manually because of rounding problems
            mapper = {
                "0": "0.1<x<1.04",
                "1": "<1.10",
                "2": "<1.19",
                "3": "<1.2",
                "4": "<1.3",
                "5": "<1.31",
                "6": "<1.46",
                "7": "<1.66",
                "8": "<2.2",
                "9": "<=19.8"
            }

        # and below is for every variable that are CAT_NLN
        else:
            # we read the previously saved quantile statistics
            # recorded from script A001
            cur_quantiles = BST_nonFloat["Quantiles"][itr - 37]
            mapper = {str(itr3): "<" + '{:.2f}'.format(cur_quantiles[itr3 + 1])
                      for itr3 in range(1, 9)}
            mapper[str(0)] = '{:.2f}'.format(
                cur_quantiles[0]) + "< x <" + '{:.2f}'.format(cur_quantiles[1])
            mapper[str(9)] = "<=" + '{:.2f}'.format(cur_quantiles[10])

        # ---
        df_all = pd.concat([df_real, df_fake], ignore_index=True)

        df_all[cur_name] = df_all[cur_name].astype(
            "int").astype(str).astype("category").map(mapper)

        plot = sns.histplot(
            x=df_all[cur_name],
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

        # the names of the xtick will be a bit long for CAT_NLN
        # so we will rotate the ticks
        # and we will move the first class slightly to the left
        if itr > 39:
            ax[i, j].tick_params(labelrotation=30)
            temp = ax[i, j].get_xticks()
            temp[0] = -0.5
            ax[i, j].set_xticks(temp)

        ax[i, j].set_title(cur_name.partition(' [')[0], fontsize=60)
        ax[i, j].set_xlabel(xlabel=None)

    plot.legend_.remove()
    ax[i, j].yaxis.set_visible(False)
    # ax[i, j].set_xlabel(xlabel=col)
    ax[i, j].tick_params(labelsize=25)

fig.tight_layout(h_pad=3, w_pad=1)
fig.delaxes(ax[8, 4])

legend_elements = [Patch(facecolor='r', edgecolor='r', label='Real Data', alpha=0.5, linewidth=3),
                   Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='%s Synthetic Data' % method, alpha=0.5, linewidth=3)]
fig.legend(handles=legend_elements, loc="lower center",
           borderaxespad=0.1, ncol=2, fontsize=45)

# plt.subplots_adjust(right=0.85)
plt.subplots_adjust(bottom=0.07)

fig.savefig("plot/%s_distributions.png" % method)
