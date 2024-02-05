import hydra
import sys
import pandas as pd
from lib.utils import save_obj
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox


@hydra.main(config_path="config", config_name="config")
def main(hp: DictConfig):
    df = pd.read_csv(hp.dir.data + "data_real.csv")

    print("Load data types...")
    dtype = {
        "index": "int32",
        "name": "str",
        "type": "str",
        "num_classes": "int32",
        "embedding_size": "int32",
        "include": "bool",
    }
    data_types = pd.read_csv(
        hp.dir.config + "data_types.csv",
        usecols=dtype.keys(),
        dtype=dtype,
        index_col="index",
    )
    data_types["index_start"] = 0
    data_types["index_end"] = 0
    id_real = data_types["type"] == "real"
    data_types = pd.concat(
        [data_types[id_real], data_types[~id_real]], ignore_index=True
    )

    print("Keep included variables...")
    included_columns = data_types.loc[data_types["include"], "name"].to_list()
    df = df[["icustay_id", "hour"] + included_columns]

    print("Transform columns...")
    current_index = 0
    transforms_dict = {}
    plotted = False
    for index, row in data_types.iterrows():
        if row["include"]:
            # real transforms
            if row["type"] == "real":
                df[row["name"]], lambda_bc = boxcox(df[row["name"]] + 1)
                minmax_scaler = MinMaxScaler(feature_range=(0, 1))
                df[row["name"]] = minmax_scaler.fit_transform(
                    df[row["name"]].values.reshape(-1, 1)
                )
                transforms_dict[row["name"]] = {
                    "lambda_bc": lambda_bc,
                    "minmax_scaler": minmax_scaler,
                }
            # plot distributions
            if row["type"] != "real" and not plotted:
                print("Plot distributions...")
                fig, ax = plt.subplots(
                    ncols=sum(data_types["include"]), nrows=1, figsize=(40, 20)
                )
                for i_plt, r_plt in data_types.iterrows():
                    if r_plt["include"]:
                        ax[i_plt].hist(df[r_plt["name"]].values, bins=100,
                                       density=True)
                        ax[i_plt].title.set_text(r_plt["name"])
                fig.savefig(hp.dir.data + "distributions.png")
                plotted = True
            # dummify categorical
            if row["type"] == "cat" or row["type"] == "bin":
                columns_before = df.columns
                df = pd.get_dummies(df, prefix=row["name"],
                                    columns=[row["name"]])
                columns_after = df.columns
                dummy_columns = [
                    x.split("_")[-1] for x in columns_after if
                    x not in columns_before
                ]
                transforms_dict[row["name"]] = {"dummy_columns": dummy_columns}
            # columns start and end indices
            data_types.at[index, "index_start"] = current_index
            current_index = current_index + row["num_classes"]
            data_types.at[index, "index_end"] = current_index

    print("Save...")
    save_obj(transforms_dict, hp.dir.data + "transforms_dict.pkl")


if __name__ == "__main__":
    # Disable logging for this script
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=disabled")
    sys.argv.append("hydra/hydra_logging=disabled")

    main()
