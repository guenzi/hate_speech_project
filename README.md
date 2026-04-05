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

## Data description:

---

## 👤 Author

```text
Loic Guenzi, ..., Lucas Firouzi
```
