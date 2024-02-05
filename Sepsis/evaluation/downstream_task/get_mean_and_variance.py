import numpy as np

# Sample data dictionary, replace this with your actual data
data = {
    # "Hypotension Fake": ,
    # "Hypotension Real": ,
    # "Hypotension Merged": ,
    # "Sepsis Fake": ,
    # "Sepsis Real": ,
    # "Sepsis Merged": ,
    # "Gender Fake": ,
    # "Gender Real": ,
    # "Gender Merged": ,
}


for data_type in data.keys():
    averages = {}
    variances = {}

    for key in data[data_type]['generation_1']:
        if key == 'SpO2':
            continue
        values = [generation[key]
                  for generation in data[data_type].values() if key in generation]
        if any([value != value for value in values]):
            continue
        averages[key] = np.mean(values)
        variances[key] = np.var(values)

    print("Data type:", data_type)
    print("Averages:", averages)
    print("Median:", np.median(list(averages.values())))
    print("Variances:", variances)
