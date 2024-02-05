import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import statistics
import datetime


parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('--testing', action='store_true', help='Enable testing mode')
parser.add_argument('--data', type=str, choices=['Fake', 'Merged', 'Real', 'RealLong'], help='Set the type of data', required=True)

args = parser.parse_args()

TESTING = args.testing
data_type = args.data
print("data_type:", data_type)

if TESTING:
    RUN_NAME = f"Gender_Runs/{data_type}/Testing Run - {datetime.datetime.now().strftime('%m-%d--%H:%M')}"
else:
    RUN_NAME = f"Gender_Runs/{data_type}/Run - {datetime.datetime.now().strftime('%m-%d--%H:%M')}"

time_series_length = 15
test_path = '../gender_sepsis_data/test_set.csv'


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def model_train(train, column, generation, model_num, order_patients, n_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BiLSTM(input_dim=1, hidden_dim=50).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for i in order_patients:
        sample = train.iloc[i * time_series_length:(i + 1) * time_series_length][column].to_list()
        train_sample, test_sample = split_sequence(sample, n_steps)

        inputs = torch.tensor(train_sample, dtype=torch.float32).unsqueeze(2).to(device)
        targets = torch.tensor(test_sample, dtype=torch.float32).unsqueeze(1).to(device)

        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model_output_path = os.path.join(RUN_NAME, f'output_{generation}', f'model_{model_num}', column)
    torch.save(model.state_dict(), model_output_path + ".pt")


def model_test(test, column, generation, model_num, n_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BiLSTM(input_dim=1, hidden_dim=50).to(device)
    model_path = os.path.join(RUN_NAME, f'output_{generation}', f'model_{model_num}', column + ".pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    differences = []
    n_patients = len(test) // time_series_length

    for i in range(n_patients):
        test_predictions = []
        for j in range(time_series_length - n_steps):
            x_input = np.array(test.iloc[i*time_series_length+j: i*time_series_length+n_steps+j][column].values.tolist())
            x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
            # print(x_input.shape)
            # print(column, i, j, n_steps)
            with torch.no_grad():
                current_pred = model(x_input).item()
            test_predictions.append(current_pred)

        true_predictions = np.array(test.iloc[i*time_series_length+n_steps:(i+1)*time_series_length][column].values.tolist())
        compare = pd.DataFrame({'Pred': test_predictions, 'Real': true_predictions})
        average = abs((compare['Pred'] - compare['Real']) / compare['Real']).mean()
        differences.append(average)

    return np.array(differences)


if __name__ == '__main__':
    generations_path = "../gender_sepsis_data/generations" if data_type != "RealLong" else "../gender_sepsis_data/generations_long"
    n_steps = 10
    n_models = 5 if not TESTING and not "Real" in data_type else 1

    all_gen_avg_errors = []

    gen_column_avg_errors = dict()  # To store average errors per column per generation
    for generation in tqdm(os.listdir(generations_path)):
        if generation == 'archive':
            continue

        if data_type == "Fake":
            file_name = "fake_data.csv"
        elif data_type == "Merged":
            file_name = "data_merged.csv"
        elif data_type == "Real":
            file_name = "real_data.csv"
        elif data_type == "RealLong":
            file_name = "real_data_long.csv"
        else:
            raise Exception("Invalid data type!")

        data_path = os.path.join(generations_path, generation, file_name)
        if "Real" in data_type:
            data_path = os.path.join(generations_path, '..', file_name)

        train = pd.read_csv(data_path)
        assert len(train) % time_series_length == 0, "invalid number of rows"

        cols_to_drop = [col for col in train.columns if '_m' in col or col in ['icustay_id', 'hour', 'ethnicity']]
        train.drop(cols_to_drop, axis='columns', inplace=True)

        n_patients = len(train) // time_series_length
        order_patients = np.arange(n_patients)
        np.random.shuffle(order_patients)

        if not os.path.exists(RUN_NAME):
            os.mkdir(RUN_NAME)

        output_gen_path = os.path.join(RUN_NAME, f'output_{generation}')
        if not os.path.exists(output_gen_path):
            os.mkdir(output_gen_path)

        column_errors = {col: [] for col in train.columns}  # Store errors per column

        for model_num in range(n_models):

            output_model_path = os.path.join(output_gen_path, f'model_{model_num}')
            if not os.path.exists(output_model_path):
                os.mkdir(output_model_path)

            for column in range(1, 35):
                model_train(train, str(column), generation, model_num, order_patients, n_steps)
                if TESTING:
                    break

        # Evaluation
        test = pd.read_csv(test_path)
        assert len(test) % time_series_length == 0, "invalid number of rows"

        cols_to_drop = [col for col in test.columns if '_m' in col]
        test.drop(cols_to_drop, axis='columns', inplace=True)

        results = pd.DataFrame()
        predictions = pd.DataFrame()

        for model_num in range(n_models):
            for column in range(1, 35):
                column = str(column)
                results[column] = model_test(test, column, generation, model_num, n_steps)
                column_errors[column].append(results[column].mean())  # Add average error for current column
                if TESTING:
                    break

            results_path = os.path.join(output_gen_path, f'model_{model_num}', 'results.csv')
            results.to_csv(results_path, index=False)

        # Calculating the average error for the column across all models
        for column in column_errors:
            column_errors[column] = np.mean(column_errors[column])

        gen_column_avg_errors[generation] = column_errors
        all_gen_avg_errors.append(np.mean(list(column_errors.values())))

        if TESTING:
            break

    # Printing and saving results
    # print("Average Error for Each Generation:", all_gen_avg_errors)
    # print("Global Average Error:", np.mean(all_gen_avg_errors))
    print("Average Errors per Column:", gen_column_avg_errors)

    average_values = []

    for key in gen_column_avg_errors['generation_1'].keys():
        sum_values = 0
        
        if pd.isna(gen_column_avg_errors['generation_1'][key]):
            continue

        for generation_data in gen_column_avg_errors.values():
            sum_values += generation_data[key]
        average_value = sum_values / len(gen_column_avg_errors)
        average_values.append(average_value)

    median_average = statistics.median(average_values)

    print("Median of Average Values:", median_average)
    print(data_type)

    # Save results to CSV
    df_column_avg_errors = pd.DataFrame(gen_column_avg_errors)
    df_column_avg_errors['Median'] = median_average
    df_column_avg_errors.to_csv(RUN_NAME + '/gen_column_avg_errors.csv', index=True)
