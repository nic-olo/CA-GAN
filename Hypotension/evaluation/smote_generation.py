import pandas as pd
from imblearn.over_sampling import SMOTE

# read file
df = pd.read_csv("data/data_real.csv")

# retrive ethnicity
demog = pd.read_csv("data/demographics_icustayid.csv")

# flattened dataframe
df_flat = df.set_index(["icustay_id", "hour"]).unstack().sort_index(1,level=1)

# merge and select ethinicities
df_flat_et = df_flat.merge(demog[["icustay_id", "ethnicity"]], on='icustay_id', how='left')

df_flat_et["ethnicity"] = df_flat_et["ethnicity"].replace({'WHITE - RUSSIAN':'WHITE', 
        'BLACK/AFRICAN':'BLACK', 
        'BLACK/AFRICAN AMERICAN':'BLACK',
        'WHITE - EASTERN EUROPEAN':'WHITE',
        'WHITE - OTHER EUROPEAN':'WHITE', 
        'WHITE - BRAZILIAN':'WHITE', 
        'BLACK/HAITIAN':'BLACK',
        'BLACK/CAPE VERDEAN':'BLACK', 
        })

df_flat_et = df_flat_et[df_flat_et.ethnicity.isin(['BLACK', 'WHITE']) == True]


## SMOTE ##

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(df_flat.loc[df_flat_et['icustay_id']], df_flat_et['ethnicity'])


# stack back into original shape
df_stack = X_res.stack()
df_stack = df_stack.reset_index()
df_stack.rename(columns = {'level_0':'id'}, inplace = True)

# add ethnicity
y_res = y_res.reset_index()
y_res.rename(columns = {'index':'id'}, inplace = True)

df_stack_et = df_stack.merge(y_res, on='id', how='left')

# add original IDs
df_stack_et.insert(1, 'icustay_id', df.set_index('icustay_id').loc[df_flat_et['icustay_id']].reset_index()['icustay_id'])

# save
df_stack_et.to_csv("data/data_smote.csv", index=False)