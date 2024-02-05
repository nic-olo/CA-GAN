import os
import pandas as pd
from tqdm import tqdm

# Path to the directory where the script will be placed and executed
base_path = os.path.dirname(os.path.realpath(__file__))
generations_folder = os.path.join(base_path, 'generations')

# Check if the directory exists
if not os.path.exists(generations_folder):
    print(f"'{generations_folder}' doesn't exist!")
    exit()

# Read the real_data.csv data
real_data_path = os.path.join(base_path, 'real_data.csv')
if not os.path.exists(real_data_path):
    print(f"'{real_data_path}' doesn't exist!")
    exit()

real_data = pd.read_csv(real_data_path)

# Loop through each subfolder in generations and process the files
for subdir, dirs, files in tqdm(os.walk(generations_folder)):
    if 'fake_data.csv' in files:
        fake_data_path = os.path.join(subdir, 'fake_data.csv')
        fake_data = pd.read_csv(fake_data_path)

        # Concatenate the real data with the fake data
        merged_data = pd.concat([real_data, fake_data])

        # Save the merged data to a new file in the same subfolder
        merged_data_path = os.path.join(subdir, 'data_merged.csv')
        merged_data.to_csv(merged_data_path, index=False)
        print(f"Merged data saved to: {merged_data_path}")

print("All done!")
