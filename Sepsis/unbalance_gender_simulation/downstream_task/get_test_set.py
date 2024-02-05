import pandas as pd
import numpy as np

df = pd.read_csv('real_data.csv')
df_long = pd.read_csv('real_data_long.csv')

original_df_long_length = len(df_long)

max_start_index = len(df_long) - 15
starting_indices = np.arange(0, max_start_index, 15)
selected_indices = np.random.choice(starting_indices, 120, replace=False)

selected_patients = pd.DataFrame()
for start_index in selected_indices:
    if len(selected_patients) >= 1200:
        break
    patient_data = df_long.iloc[start_index:start_index + 15]

    patient_data = patient_data[patient_data['0'] <= 91]

    if patient_data.empty or patient_data['0'].iloc[0] in df['0'].values:
        continue

    assert all(patient_data['0'] == patient_data['0'].iloc[0])

    selected_patients = pd.concat([selected_patients, patient_data], ignore_index=True)

    df_long = df_long[df_long['0'] != patient_data['0'].iloc[0]]


assert len(df_long) == original_df_long_length - len(selected_patients)

selected_patients.to_csv('test_set.csv', index=False)
df_long.to_csv('real_data_long.csv', index=False)
print(selected_patients)
