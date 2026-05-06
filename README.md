# Detecting Racism on Twitter using Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)

Detection of racism in tweets using BERTweet, with a focus on inter-group bias: does the model perform equally well across all ethnic groups? We study this bias and mitigate it via weighted loss and data augmentation.

---

## Setup

Clone the repository locally and on the cluster:

```bash
git clone https://github.com/guenzi/hate_speech_project.git
cd hate_speech_project
```

The pipeline runs on the **EPFL RCP cluster via RunAI** using a pre-built Docker image that includes all dependencies. You do not need to install anything locally.

Docker image: `registry.rcp.epfl.ch/ee-559-guenzi/my-toolbox:v0.2`

The `requirements.txt` lists the packages included in the image for reference only.

---

## Repository Structure

```
src/
  preprocess.py     — tweet cleaning and label generation from HateXplain
  config.py         — shared constants (GROUPS, MODEL_NAME, SEED, get_device)
  dataset.py        — TweetDataset, make_loaders, load_data, split_data
  model.py          — FocalLoss, train_bertweet, predict, save_checkpoint
  evaluate.py       — evaluate_per_group, compute_sample_weights
  augment.py        — data augmentation (synonym, deletion, swap, noise)
  plots.py          — training curves and group F1 comparison plots

main.py             — pipeline orchestrator
requirements.txt    — dependencies
old_architecture/   — old files kept for reference
```

---

## Data

`data/HateXplain.json` is tracked by git and does not need to be downloaded separately.

Only mono-ethnic posts are kept (posts targeting exactly one racial group), which ensures clean per-group evaluation.

| Dataset | Source | Role | Label |
|---|---|---|---|
| **HateXplain** | [GitHub](https://github.com/hate-alert/HateXplain) | Train / Val / Test | hatespeech + racial target → 1, normal → 0 |

---

## How to Run

### Step 1 — Preprocess (once, or after any change to preprocess.py)

```bash
cd src
python3 preprocess.py
```

Generates `data/final_datasets/hatexplain.csv`. Skipped automatically if the file already exists.

### Step 2 — Full pipeline

```bash
python3 main.py \
  --data_path data/final_datasets/hatexplain.csv \
  --output_dir results \
  --epochs 5 \
  --patience 2
```

Each run creates a timestamped subdirectory (e.g. `results/20260506_143012/`) so previous results are never overwritten.

### Pipeline steps

1. **SVM + TF-IDF** — baseline
2. **BERTweet Baseline** — fine-tuned BERTweet, no balancing
3. **BERTweet Weighted** — focal loss + per-group sample weights
4. **BERTweet Weighted + Aug** — same + data augmentation on underrepresented groups

### Output per run

```
results/<timestamp>/
  results.json                        — global and per-group F1 for all models
  bertweet_baseline_curves.png
  bertweet_weighted_curves.png
  bertweet_weighted_plus_aug_curves.png
  group_f1_comparison.png             — all models compared by ethnic group
  augmentation_report.json
  train_augmented.csv
  checkpoints/                        — model weights (gitignored)
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 5 | Max training epochs |
| `--patience` | 2 | Early stopping patience |
| `--batch_size` | 16 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max_len` | 128 | Max token length |
| `--aug_factor` | 2 | Augmented variants per sample |
| `--aug_min_group_samples` | 700 | Threshold below which a group is augmented |
| `--aug_methods` | synonym,delete,swap,punct,char,combo | Augmentation methods |
| `--augment_labels` | positive | Augment only label=1 (or "all") |

---

## Running on the EPFL Cluster (RunAI)

Replace `<username>` with your GASPAR username and `<uid>` with your UID (run `id` on the jumphost to find it).

Connect via VPN (`vpn.epfl.ch`) then SSH:

```bash
ssh <username>@jumphost.rcp.epfl.ch
```

Pull the latest code:

```bash
cd ~/hate_speech_project
git pull
```

### Preprocess job

```bash
runai submit job-preprocess \
  --run-as-uid <uid> \
  --image registry.rcp.epfl.ch/ee-559-guenzi/my-toolbox:v0.2 \
  --existing-pvc claimname=course-ee-559-scratch-g17,path=/scratch \
  --existing-pvc claimname=home,path=/home/<username> \
  --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
  --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
  --command -- bash -c "cd /home/<username>/hate_speech_project/src && python3 preprocess.py"
```

### Training job

```bash
runai submit job-racism-vX \
  --run-as-uid <uid> \
  --image registry.rcp.epfl.ch/ee-559-guenzi/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=home,path=/home/<username> \
  --command -- python3 /home/<username>/hate_speech_project/main.py \
    --data_path /home/<username>/hate_speech_project/data/final_datasets/hatexplain.csv \
    --output_dir /home/<username>/hate_speech_project/results \
    --epochs 5 --patience 2
```

Replace `vX` with a unique number (v5, v6…).

### Useful commands

```bash
runai list
runai logs <job-name> -f
runai delete job <job-name> -p course-ee-559-<username>
```

### Retrieve results locally (VPN required)

```bash
scp -r <username>@jumphost.rcp.epfl.ch:/home/<username>/hate_speech_project/results/ \
    "/path/to/local/hate_speech_project/"
```

---

## Authors

Loïc Guenzi, Lucas Firouzi, Rémy Jaillat
