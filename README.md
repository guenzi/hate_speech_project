# Detecting Racism on Twitter using Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)

This repository contains the code for the detection of racism in twitter commentary and messages using deep learning.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/guenzi/hate_speech_project.git
cd hate_speech_project
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional: verify that required versions are installed

---

## Repository Structure

- `data/`: Store the data used in the project
- `notebooks/`: Store the notebooks used for testing during the project
  - `Test.ipynb`: First test notebook
- `src/`: Contain the code for the pipeline
    - `preprocess.py`: Tweet cleaning and label formatting
    - `dataset.py`: PyTorch Dataset class
    - `model.py`: Definition of the BERTweet model architecture
    - `train.py`: Training and validation loop
    - `tools.py`: Additional helper
- `README.md`: README file
- `main.py`: Script to run the whole pipeline
- `requirements.txt`: Requirements to run the code in this repository

---

## Data description

Due to file size limitations, datasets are not included in this repository.

Download the data from the following Google Drive link:
https://drive.google.com/drive/folders/1KJfLX84MFirhX5qh1UByg9MwIB8_2_o-?usp=sharing

Then, manually create a `data/` folder at the root of the project and place the downloaded files inside.

### Datasets

| Dataset | Source | Role | Label used |
|---|---|---|---|
| **HateXplain** | [GitHub](https://github.com/hate-alert/HateXplain) | Train / Val | `hatespeech` + racial target → 1, `normal` → 0 |
| **Implicit Hate Corpus** | [GitHub](https://github.com/SALT-NLP/implicit-hate) | Test (implicit) | stg2 implicit posts → 1, stg1 `not_hate` → 0 |
| **CAD** | [Zenodo](https://doi.org/10.5281/zenodo.4881008) | Bonus / cross-domain | `IdentityDirectedAbuse` → 1, `Neutral` → 0 |
| **Gab Hate Corpus** | [OSF](https://osf.io/edua3/) | Bonus / cross-domain | `hd=1` → 1, `hd=0` → 0 |

All datasets are preprocessed by `src/preprocess.py` and saved as CSVs in `data/final_datasets/`.

---

## 👤 Author

```text
Loïc Guenzi, Lucas Firouzi, Rémy Jaillat
```
