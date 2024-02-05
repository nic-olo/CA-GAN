# Conditional Augmentation GAN (CA-GAN)

This is the repository for the Conditional Augmentation GAN (CA-GAN).

Details about the project can be found in our paper "Generative AI Mitigates Representation Bias Using Synthetic Health Data".

Link to the paper: [Generative AI Mitigates Representation Bias Using Synthetic Health Data
](https://www.medrxiv.org/content/10.1101/2023.09.26.23296163v2)

Link to our previous work on CA-GAN: [[2210.13958] Mitigating Health Data Poverty: Generative Approaches versus Resampling for Time-series Clinical Data](https://arxiv.org/abs/2210.13958)


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


## Data Access

The data for our project came from the MIMIC-III database. We cannot provide the data directly, but access to the database can be requested through PhysioNet (https://physionet.org/content/mimiciii/1.4/).

We chose two datasets extracted from the MIMIC-III database. The inclusion and exclusion criteria are described in the literature:

- Acute hypotension: [Interpretable Off-Policy Evaluation in Reinforcement Learning by Highlighting Influential Transitions](https://proceedings.mlr.press/v119/gottesman20a.html) [GitHub - dtak/interpretable_ope_public](https://github.com/dtak/interpretable_ope_public)
  
- Sepsis: [The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care | Nature Medicine](https://www.nature.com/articles/s41591-018-0213-5) [AIClinician / AIClinician · GitLab](https://gitlab.doc.ic.ac.uk/AIClinician/AIClinician)
  

To demo this code without access to Physionet, please refer to Appendix B of our [paper](https://www.medrxiv.org/content/10.1101/2023.09.26.23296163v2) to recreate a sample dataset.

## Usage

Get a copy of this project and set it up on your local machine.

Install the required packages (e.g. `pip install -r requirements.txt`).

The folders "Hypotension" and "Sepsis" are independent of each other. They represent the two case studies on which we applied CA-GAN. Both folders contain the code needed to:

- retrieve and preprocess data from the MIMIC-III database to obtain the real data used to train our model.
- train CA-GAN on the real data
- use the trained model to generate the synthetic data
- evaluate the quality of the data generated

## Support

For support or inquiries, please contact nmicheletti@fbk.eu and rmarchesi@fbk.eu

## Acknowledgment

This project is based on *Kuo, Nicholas I-Hsien, et al. "The Health Gym: synthetic health-related datasets for the development of reinforcement learning algorithms." Scientific Data 9.1 (2022): 693.* (Copyright (c) 2022. by Nicholas Kuo & Sebastiano Babieri, UNSW.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
