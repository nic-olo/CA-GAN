# Sepsis

This is the folder for the Sepsis dataset.

The structure of the "Sepsis" folder is organized as follows:

1. real_data_extraction
2. synthetic_data_generation
3. evaluation
4. unbalance_gender_simulation

## Real Data Extraction

The "real_data_extraction" subfolder contains Jupyter Notebooks for extracting specific data related to sepsis from MIMIC-III database. To create the real sepsis dataset, follow this steps:

1. `AIClinician_Data_extract_MIMIC3_BigQuery.ipynb`: Notebook for extracting data using MIMIC-III on BigQuery.
2. `sepsis_def.ipynb`
3. `mimic3_dataset.ipynb`
4. `Core.ipynb`
5. `ethnicity_merge.ipynb`

We cannot provide the data directly, but access to the database can be requested through PhysioNet ([MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)).

The inclusion and exclusion criteria are described in the literature: [The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care | Nature Medicine](https://www.nature.com/articles/s41591-018-0213-5) [AIClinician / AIClinician · GitLab](https://gitlab.doc.ic.ac.uk/AIClinician/AIClinician)

This code comes from [GitHub - uribyul/py_ai_clinician](https://github.com/uribyul/py_ai_clinician) and it is the Python version of the original Matlab implementation.

To demo this code without access to Physionet, please refer to Appendix B of our [paper](https://www.medrxiv.org/content/10.1101/2023.09.26.23296163v2https://www.medrxiv.org/content/10.1101/2023.09.26.23296163v2) to recreate a sample dataset.

## Synthetic Data Generation

The "synthetic_data_generation" subfolder is dedicated to scripts for generating synthetic sepsis data:

- `C007_Utils_BackTransform.py`: Script for back transformation.
- `C008_Utils_ReplaceNames.py`: Script for replacing names.
- `disclosure.py`: Script for data disclosure.
- `plots.py`: Script for generating plots.
- `prepare_real_data.py`: Script for preparing real data.
- `save_results.py`: Script for generatin synthetic data.
- `utils_loaders.py`: Script containing utility loaders.
- `ca-gan_train.py`: Script for training CA-GAN models.
- Additional scripts in the subfolders:
  - `config.yaml`
  - `utils.py`
  - `ca-gan.py`

To generate synthetic data, follow these steps:

1. `prepare_real_data.py`
  
2. `wgan_train.py`
  
3. `save_results.py`
  

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
  - `sepsis_train_LSTM.py`
  - `merge_real.py`

## Unbalanced Gender Simulation

The "unbalance_gender_simulation" subfolder contains scripts and notebooks for simulating an unbalanced gender scenario and repeat the Sepsis experiment under this condition.

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
