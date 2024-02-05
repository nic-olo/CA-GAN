# ===>>>
# This is the 1st of all files for WGAN on Sepsis

# ===>>>
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import yaml

# ===>>>
# (Part 1)      Load the data
data_dir = "../data/"
file_name = "mimictable15.csv"

MyData = pd.read_csv(data_dir + file_name)

length = len(MyData[MyData['icustayid'] == MyData['icustayid'].unique()[0]])

with open("./config/config.yaml") as f:
    list_doc = yaml.safe_load(f)

for sense in list_doc:
    if sense == "max_sequence_length":
        list_doc[sense] = length

with open("./config/config.yaml", "w") as f:
    yaml.dump(list_doc, f)

# ===>>>
# (Part 2)      Creating empty data frames for
#               finalising the data

# ---
# (Part 2.1)    The following is for documenting
#               features of the data
MyData_Types = pd.DataFrame()

MyData_Types["index"] = []
MyData_Types["name"] = []
MyData_Types["type"] = []
MyData_Types["num_classes"] = []
MyData_Types["embedding_size"] = []
MyData_Types["include"] = []
MyData_Types["index_start"] = []
MyData_Types["index_end"] = []

# ---
# (Part 2.2)    The following is for storing
#               the transformed dataset
MyData_Transformed = pd.DataFrame()

MyData_Transformed['Admn001_ID'] = MyData['icustayid']

# ===>>>
# (Part 3)      Below, we sort out different types
#               of variables

# ---
# (Part 3.1)    Float variables (FLT),
#               those that are "normal" normal (N2) &
#               those that are log normal (LN)
Flt_Variable_N2 = ["age",
                   "HR", "SysBP", "MeanBP",
                   "DiaBP", "RR", "SpO2",
                   "Potassium", "Sodium", "Chloride",
                   "Calcium", "Ionised_Ca", "CO2_mEqL",
                   "Albumin", "Hb", "Arterial_pH",
                   "Arterial_BE", "HCO3",
                   "FiO2_1"
                   ]

Flt_Variable_LN = ["Glucose", "BUN", "Creatinine",
                   "Magnesium", "SGOT", "SGPT",
                   "Total_bili", "WBC_count",
                   "Platelets_count", "paO2", "paCO2",
                   "Arterial_lactate",
                   "input_total", "input_4hourly", "max_dose_vaso",
                   "output_total", "output_4hourly"
                   ]

# ---
# (Part 3.2)    Binary variables (Bin)
Bin_Variable = ["gender", "re_admission",
                "mechvent"
                ]

Cat_Variable_TEMP = ["Temp_C"
                     ]

# ---
# (Part 3.3)    Categorical variables (Cat),
#               those that are multi-classes (MTC) &
#               those that are not log normal (NLN)
Cat_Variable_MTC = ["GCS"
                    ]

# Note, the following variables are actually floats in nature however,
# they have long tails and do not follow log normalisation nicely To address
# this problem, we will simply treat them as quasi-categorical variables
Cat_Variable_NLN = ["SpO2", "PTT", "PT",
                    "INR"
                    ]

# ===>>>
# (Part 4)      Storing some important
#               back-transformation statistics (BTS) down the road
A001_BTS_Float = {"Name": [], "min_X0": [], "max_X1": [], "LogNormal": []}

A001_BTS_nonFloat = {"Name": [], "Type": [], "Quantiles": []}

# ===>>>
# (Part 5)      Sorting the data

# Minmax will be used for our data transformation
minmax_scaler = MinMaxScaler()

# ---
# (Part 5.1)    Sorting out floats with normal distribution
for itr in range(len(Flt_Variable_N2)):
    # ---
    if itr == 0:
        Cur_Types_Row = 0
        Cur_Index_Row = 0
    else:
        Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    # ---
    Cur_Name = Flt_Variable_N2[itr]
    Cur_Val = np.array(list(MyData[Cur_Name]))

    # ---
    # Appending the data type data frame
    MyData_Types = MyData_Types. \
        append({'index': Cur_Index_Row,
                'name': Cur_Name,
                'type': 'real',
                'num_classes': 1,
                'embedding_size': 1,
                'include': True,
                'index_start': Cur_Types_Row,
                'index_end': Cur_Types_Row + 1
                },
               ignore_index=True
               )
    # ---
    # Documenting the BTS
    A001_BTS_Float["Name"].append(Cur_Name)

    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    A001_BTS_Float["LogNormal"].append(False)

    # ---
    # Appending the transformed data frame
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val

    # ---
    Cur_Index_Row += 1

# ---
# (Part 5.2)    Sorting out floats with log distribution
for itr in range(len(Flt_Variable_LN)):
    # ---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    # ---
    Cur_Name = Flt_Variable_LN[itr]
    Cur_Val = np.array(list(MyData[Cur_Name]))

    Cur_Val = np.log(Cur_Val + 1)

    # ---
    MyData_Types = MyData_Types. \
        append({'index': Cur_Index_Row,
                'name': Flt_Variable_LN[itr],
                'type': 'real',
                'num_classes': 1,
                'embedding_size': 1,
                'include': True,
                'index_start': Cur_Types_Row,
                'index_end': Cur_Types_Row + 1
                },
               ignore_index=True
               )
    # ---
    A001_BTS_Float["Name"].append(Cur_Name)

    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    A001_BTS_Float["LogNormal"].append(True)

    # ---
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val

    # ---
    Cur_Index_Row += 1

# ---
# (Part 5.3)    Sorting out the binary variables
for itr in range(len(Bin_Variable)):
    # ---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    # --
    Cur_Name = Bin_Variable[itr]
    Cur_Val = np.array(list(MyData[Cur_Name]))

    # ---
    MyData_Types = MyData_Types. \
        append({'index': Cur_Index_Row,
                'name': Bin_Variable[itr],
                'type': 'bin',
                'num_classes': 2,
                'embedding_size': 2,
                'include': True,
                'index_start': Cur_Types_Row,
                'index_end': Cur_Types_Row + 2
                },
               ignore_index=True
               )
    # ---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("bin")

    # only CAT_NLNs use quantiles
    A001_BTS_nonFloat["Quantiles"].append({})

    # ---
    # store the classes explicitly in separate columns
    # because we will be using "soft embedding" later in our WGAN
    for itr2 in range(2):
        Temp_Name = Cur_Name + '_' + str(itr2)
        Temp_Val = np.zeros_like(Cur_Val)

        Loc_Ele = np.where(Cur_Val == itr2)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]). \
            astype(int)

    # ---
    Cur_Index_Row += 1

# ---
# (Part 5.5)    Sorting out the Temp (C) variables
for itr in range(len(Cat_Variable_TEMP)):
    # ---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    # ---
    Cur_Name = Cat_Variable_TEMP[itr]
    Cur_Val = np.floor(np.array(list(MyData[Cur_Name])))

    # ---
    MyData_Types = MyData_Types. \
        append({'index': Cur_Index_Row,
                'name': Cat_Variable_TEMP[itr],
                'type': 'cat',
                'num_classes': 3,
                'embedding_size': 3,
                'include': True,
                'index_start': Cur_Types_Row,
                'index_end': Cur_Types_Row + 3
                },
               ignore_index=True
               )
    # ---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("Temp_C")
    A001_BTS_nonFloat["Quantiles"].append([0.0, 35.05, 38, 50.0])

    quantiles = [0.0, 35.05, 38, 50.0]
    # ---
    for itr2 in range(3):
        Temp_Name = Cur_Name + '_C' + str(itr2)
        Temp_Val = np.zeros_like(Cur_Val)

        Lower_bar = quantiles[itr2]
        Upper_bar = quantiles[itr2+1]

        # technical little difficulty
        if itr2 == 2:
            Upper_bar = Upper_bar * 1.05

        # find those desired variables by index
        Loc_Ele = np.all([[Cur_Val >= Lower_bar],
                          [Cur_Val < Upper_bar]],
                         axis=0)[0]
        Temp_Val[Loc_Ele] = 1
        Loc_Ele = np.where(Cur_Val == itr2)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]). \
            astype(int)

    # ---
    Cur_Index_Row += 1

# ---
# (Part 5.4)    Sorting out the MTC (GCS) variables
for itr in range(len(Cat_Variable_MTC)):
    # ---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    # ---
    Cur_Name = Cat_Variable_MTC[itr]
    Cur_Val = np.floor(np.array(list(MyData[Cur_Name])))

    # ---
    MyData_Types = MyData_Types. \
        append({'index': Cur_Index_Row,
                'name': Cat_Variable_MTC[itr],
                'type': 'cat',
                'num_classes': 13,
                'embedding_size': 4,
                'include': True,
                'index_start': Cur_Types_Row,
                'index_end': Cur_Types_Row + 13
                },
               ignore_index=True
               )
    # ---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("GCS")
    A001_BTS_nonFloat["Quantiles"].append({})

    # ---
    for itr2 in range(3, 16):
        Temp_Name = Cur_Name + '_' + str(itr2)
        Temp_Val = np.zeros_like(Cur_Val)

        Loc_Ele = np.where(Cur_Val == itr2)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]). \
            astype(int)

    # ---
    Cur_Index_Row += 1

# ---
# (Part 5.5)    Sorting out the NLN variable

# we will first define how many classes do we want for the NLNs
NLN_classes = 10

# ---
for itr in range(len(Cat_Variable_NLN)):
    # ---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    # ---
    Cur_Name = Cat_Variable_NLN[itr]
    Cur_Val = np.array(list(MyData[Cur_Name]))

    # ---
    MyData_Types = MyData_Types. \
        append({'index': Cur_Index_Row,
                'name': Cat_Variable_NLN[itr],
                'type': 'cat',
                'num_classes': NLN_classes,
                'embedding_size': 4,
                'include': True,
                'index_start': Cur_Types_Row,
                'index_end': Cur_Types_Row + NLN_classes
                },
               ignore_index=True
               )
    # ---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("cat")
    A001_BTS_nonFloat["Quantiles"].append(
        [np.quantile(Cur_Val, i / NLN_classes) for i in
         range(NLN_classes + 1)])

    # ---
    for itr2 in range(NLN_classes):
        Temp_Name = Cur_Name + '_C' + str(itr2)
        Temp_Val = np.zeros_like(Cur_Val)

        # let's find the lower and the upper bounds of the 2 quantiles of
        # interest
        Lower_bar = np.quantile(Cur_Val, itr2 / NLN_classes)
        Upper_bar = np.quantile(Cur_Val, (itr2 + 1) / NLN_classes)

        # technical little difficulty
        if itr2 == (NLN_classes - 1):
            Upper_bar = Upper_bar * 1.05

        # find those desired variables by index
        Loc_Ele = np.all([[Cur_Val >= Lower_bar],
                          [Cur_Val < Upper_bar]],
                         axis=0)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]). \
            astype(int)
    # ---
    Cur_Index_Row += 1

# ===>>>
# (Part 6)      Final type check and save

# ---
torch.save(A001_BTS_Float, data_dir + 'A001_BTS_Float')

torch.save(A001_BTS_nonFloat, data_dir + 'A001_BTS_nonFloat')
# ---
MyData_Types['index'] = (MyData_Types['index']).astype(int)
MyData_Types['name'] = (MyData_Types['name']).astype(str)
MyData_Types['type'] = (MyData_Types['type']).astype(str)
MyData_Types['num_classes'] = (MyData_Types['num_classes']).astype(int)
MyData_Types['embedding_size'] = (MyData_Types['embedding_size']).astype(int)
MyData_Types['include'] = (MyData_Types['include']).astype(bool)
MyData_Types['index_start'] = (MyData_Types['index_start']).astype(int)
MyData_Types['index_end'] = (MyData_Types['index_end']).astype(int)

MyData_Types.to_csv(data_dir + 'A001_data_types.csv', index=False)

# ---
patients = dict()
for patient in MyData_Transformed['Admn001_ID']:
    if patient in patients.keys():
        patients[patient] += 1
    else:
        patients[patient] = 1

keys = []
for key in patients.keys():
    if patients[key] < length:
        keys.append(key)

MyData_Transformed = MyData_Transformed.drop(MyData_Transformed[
    MyData_Transformed[
        'Admn001_ID'].isin(
        keys)].index,
    inplace=False
)

MyData_Transformed.to_csv(data_dir + 'A001_data_real_transformed.csv',
                          index=False)
MyData_Transformed.to_pickle(data_dir + 'A001_data_real_transformed.pkl')
