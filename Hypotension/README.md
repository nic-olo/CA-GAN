# Hypotension

This is the folder for the Acute Hypotension dataset.

The structure of the "Hypotension" folder is organized as follows:

1. real_data_extraction
2. real_data_preprocessing
3. synthetic_data_generation
4. evaluation

## Real Data Extraction

The "real_data_extraction" subfolder contains SQL scripts for extracting specific data related to hypotension from MIMIC-III database:

- `charts.sql`
- `fluid_boluses.sql`
- `outputs.sql`
- `select_icu_stays.sql`
- `vasopressors.sql`

We cannot provide the data directly, but access to the database can be requested through PhysioNet ([MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)).

The inclusion and exclusion criteria are described in the literature: [Interpretable Off-Policy Evaluation in Reinforcement Learning by Highlighting Influential Transitions](https://proceedings.mlr.press/v119/gottesman20a.html) [GitHub - dtak/interpretable_ope_public](https://github.com/dtak/interpretable_ope_public).

To demo this code without access to Physionet, please refer to Appendix B of our [paper](https://www.medrxiv.org/content/10.1101/2023.09.26.23296163v2https://www.medrxiv.org/content/10.1101/2023.09.26.23296163v2) to recreate a sample dataset.

## Real Data Preprocessing

The "real_data_preprocessing" subfolder includes scripts for preprocessing real hypotension data:

- `charts.py`
- `combine_all.py`: Script for combining all data.
- `fluid_boluses.py`
- `outputs.py`
- `vasopressors.py`

## Synthetic Data Generation

The "synthetic_data_generation" subfolder is dedicated to scripts for generating synthetic data related to hypotension:

- `disclosure.py`: Script for data disclosure.
- `make_dict.py`: Script for creating dictionaries.
- `plots.py`: Script for generating plots.
- `prepare_real_data.py`: Script for preparing real data.
- `ca-gan_generate_data.py`: Script for generating data using CA-GAN.
- `ca-gan_train.py`: Script for training CA-GAN models.
- Additional scripts in the subfolders:
  - `config.yaml`
  - `utils.py`
  - `ca-gan.py`

To generate synthetic data, follow these steps:

1. `prepare_real_data.py`
  
2. `ca-gan_train.py`
  
3. `ca-gan_generate_data.py`
  

## Evaluation

The "evaluation" subfolder contains scripts and notebooks for evaluating hypotension data:

- `kendall_correlations.py`: Script for calculating Kendall correlations.
- `kl_divergence.py`: Script for calculating Kullback-Leibler divergence.
- `mmd.py`: Script for computing Maximum Mean Discrepancy.
- `plots_distributions.py`: Script for generating plots related to data distributions.
- `smote_generation.py`: Script for generating synthetic data using SMOTE.
- `tsne_pca_umap.ipynb`: Jupyter Notebook for visualizing data using t-SNE, PCA, and UMAP.
- Additional scripts in the "downstream_task" subfolder to train and test an LSTM on the data:
  - `get_mean_and_variance.py`
  - `hypotension_train_evaluate_LSTM.py`
  - `merge_real.py`

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

## Support

For support or inquiries, please contact nmicheletti@fbk.eu and rmarchesi@fbk.eu

If you use this repository, please cite:

```plaintext
@article{micheletti2023generative,
 title={Generative AI Mitigates Representation Bias Using Synthetic Health Data},
 author={Micheletti, Nicolo and Marchesi, Raffaele and Kuo, Nicholas I-Hsien and Barbieri, Sebastiano and Jurman, Giuseppe and Osmani, Venet},
 journal={medRxiv},
 pages={2023--09},
 year={2023},
 publisher={Cold Spring Harbor Laboratory Press}
}
```
