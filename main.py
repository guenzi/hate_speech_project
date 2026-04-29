"""
Racism Detection Pipeline — Cluster version
Usage:
    python3 main.py --data_path /path/to/hatexplain.csv --output_dir /path/to/results/
"""

import argparse
import ast
import json
import os
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
GROUPS     = ["African", "Asian", "Jewish", "Arab", "Caucasian", "Hispanic", "Indian", "Islam"]
MODEL_NAME = "vinai/bertweet-base"
SEED       = 42


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_per_group(df_eval, y_pred):
    results = {}
    y_pred = np.array(y_pred)
    for group in GROUPS:
        mask_pos = df_eval["targets_parsed"].apply(lambda t: group in t) & (df_eval["label"] == 1)
        mask_neg = df_eval["label"] == 0
        mask     = mask_pos | mask_neg
        if mask_pos.sum() < 5:
            continue
        results[group] = {
            "f1"   : f1_score(df_eval[mask]["label"].values, y_pred[mask], pos_label=1, zero_division=0),
            "n_pos": int(mask_pos.sum()),
        }
    return results


def print_group_results(results, model_name=""):
    print(f"\n--- F1 par groupe ({model_name}) ---")
    rows = [{"Groupe": g, "F1": round(v["f1"], 3), "N positifs": v["n_pos"]}
            for g, v in sorted(results.items(), key=lambda x: -x[1]["f1"])]
    print(pd.DataFrame(rows).to_string(index=False))


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved → {path}")


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data(data_path):
    df = pd.read_csv(data_path)
    df["targets_parsed"] = df["targets"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return df


def split_data(df):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df_train, df_temp = train_test_split(df,      test_size=0.30, random_state=SEED, stratify=df["label"])
    df_val,   df_test = train_test_split(df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["label"])
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


# ── Dataset ────────────────────────────────────────────────────────────────────
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, sample_weights=None):
        self.texts          = list(texts)
        self.labels         = list(labels)
        self.tokenizer      = tokenizer
        self.max_len        = max_len
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc  = self.tokenizer(self.texts[idx], max_length=self.max_len,
                              padding="max_length", truncation=True, return_tensors="pt")
        item = {"input_ids"     : enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label"         : torch.tensor(self.labels[idx], dtype=torch.long)}
        if self.sample_weights is not None:
            item["weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return item


def make_loaders(df_tr, df_v, df_te, tokenizer, max_len, batch_size, sample_weights=None):
    return (
        DataLoader(TweetDataset(df_tr["text"], df_tr["label"], tokenizer, max_len, sample_weights),
                   batch_size=batch_size, shuffle=True),
        DataLoader(TweetDataset(df_v["text"],  df_v["label"],  tokenizer, max_len),
                   batch_size=batch_size),
        DataLoader(TweetDataset(df_te["text"], df_te["label"], tokenizer, max_len),
                   batch_size=batch_size),
    )


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss — pénalise davantage les exemples difficiles/mal classifiés."""
    def __init__(self, gamma=2.0, reduction="none"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        ce   = nn.functional.cross_entropy(logits, labels, reduction="none")
        pt   = torch.exp(-ce)                        # probabilité de la bonne classe
        loss = (1 - pt) ** self.gamma * ce           # down-weight les exemples faciles
        if self.reduction == "mean":
            return loss.mean()
        return loss


# ── Training ───────────────────────────────────────────────────────────────────
def train_bertweet(train_loader, val_loader, device, epochs, lr,
                   use_sample_weights=False, patience=2):
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    focal_loss = FocalLoss(gamma=2.0, reduction="none")
    history    = {"train_loss": [], "val_f1": []}

    best_val_f1    = 0.0
    best_state     = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            loss   = focal_loss(logits, batch["label"].to(device))
            if use_sample_weights and "weight" in batch:
                loss = (loss * batch["weight"].to(device)).mean()
            else:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device)).logits
                preds.extend(logits.argmax(-1).cpu().numpy())
                trues.extend(batch["label"].numpy())

        val_f1   = f1_score(trues, preds, average="macro")
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  ✓ Meilleur modèle sauvegardé (val F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ✗ Pas d'amélioration ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"  Early stopping à l'epoch {epoch+1}")
                break

    # Restaurer le meilleur état
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, history


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(-1).cpu().numpy())
    return np.array(preds)


def compute_sample_weights(df_train):
    group_counts  = {g: max(1, df_train["targets_parsed"].apply(lambda t: g in t).sum()) for g in GROUPS}
    max_count     = max(group_counts.values())
    group_weights = {g: max_count / c for g, c in group_counts.items()}

    print("\nPoids par groupe :")
    for g, w in sorted(group_weights.items(), key=lambda x: -x[1]):
        print(f"  {g:<12} count={group_counts[g]:4d}  weight={w:.2f}")

    def get_weight(row):
        if row["label"] == 0:
            return 1.0
        targets = [t for t in row["targets_parsed"] if t in group_weights]
        return max((group_weights[t] for t in targets), default=1.0)

    return df_train.apply(get_weight, axis=1).values


# ── Plots ──────────────────────────────────────────────────────────────────────
def save_plots(history_base, history_weighted, group_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_base["train_loss"],     label="Baseline", color="tomato")
    axes[0].plot(history_weighted["train_loss"], label="Weighted", color="seagreen")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_base["val_f1"],     label="Baseline", color="tomato")
    axes[1].plot(history_weighted["val_f1"], label="Weighted", color="seagreen")
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()

    # Per-group F1 comparison
    svm_res, base_res, weighted_res = group_results
    all_groups = sorted(set(base_res) & set(weighted_res) & set(svm_res))
    x, width   = np.arange(len(all_groups)), 0.25
    fig, ax    = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, [svm_res[g]["f1"]      for g in all_groups], width, label="SVM",              color="steelblue", alpha=0.85)
    ax.bar(x,         [base_res[g]["f1"]      for g in all_groups], width, label="BERTweet Baseline", color="tomato",    alpha=0.85)
    ax.bar(x + width, [weighted_res[g]["f1"]  for g in all_groups], width, label="BERTweet Weighted", color="seagreen",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 par groupe ethnique ciblé")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_f1_comparison.png"), dpi=150)
    plt.close()
    print(f"\nGraphiques sauvegardés dans {output_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device = get_device()
    print(f"Device : {device}")

    # Sous-dossier horodaté pour ne pas écraser les anciens résultats
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Run ID : {run_id}  →  résultats dans {args.output_dir}")

    # Data
    print("\n[1/7] Chargement des données...")
    df = load_data(args.data_path)
    print(f"Total : {len(df)} | Racist: {(df['label']==1).sum()} | Normal: {(df['label']==0).sum()}")
    df_train, df_val, df_test = split_data(df)
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # SVM baseline
    print("\n[2/7] Baseline SVM + TF-IDF...")
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf",   LinearSVC(C=1.0, max_iter=2000, random_state=SEED)),
    ])
    svm_pipe.fit(df_train["text"], df_train["label"])
    y_pred_svm    = svm_pipe.predict(df_test["text"])
    svm_f1_global = f1_score(df_test["label"], y_pred_svm, average="macro")
    print(classification_report(df_test["label"], y_pred_svm, target_names=["Not Racist", "Racist"]))
    svm_group_results = evaluate_per_group(df_test, y_pred_svm)
    print_group_results(svm_group_results, "SVM")

    # BERTweet setup
    print("\n[3/7] Chargement tokenizer BERTweet...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    train_loader, val_loader, test_loader = make_loaders(
        df_train, df_val, df_test, tokenizer, args.max_len, args.batch_size
    )

    # BERTweet baseline
    print("\n[4/7] Entraînement BERTweet baseline...")
    model_base, history_base = train_bertweet(
        train_loader, val_loader, device, args.epochs, args.lr,
        use_sample_weights=False, patience=args.patience
    )
    save_checkpoint(model_base, os.path.join(args.output_dir, "checkpoints", "bertweet_baseline.pt"))

    y_pred_base    = predict(model_base, test_loader, device)
    base_f1_global = f1_score(df_test["label"], y_pred_base, average="macro")
    print("\n=== BERTweet Baseline ===")
    print(classification_report(df_test["label"], y_pred_base, target_names=["Not Racist", "Racist"]))
    base_group_results = evaluate_per_group(df_test, y_pred_base)
    print_group_results(base_group_results, "BERTweet Baseline")

    # Weighted loss
    print("\n[5/7] Calcul des poids par groupe...")
    sample_weights = compute_sample_weights(df_train)
    train_loader_w, _, _ = make_loaders(
        df_train, df_val, df_test, tokenizer, args.max_len, args.batch_size, sample_weights
    )

    print("\n[6/7] Entraînement BERTweet weighted...")
    model_weighted, history_weighted = train_bertweet(
        train_loader_w, val_loader, device, args.epochs, args.lr,
        use_sample_weights=True, patience=args.patience
    )
    save_checkpoint(model_weighted, os.path.join(args.output_dir, "checkpoints", "bertweet_weighted.pt"))

    y_pred_weighted    = predict(model_weighted, test_loader, device)
    weighted_f1_global = f1_score(df_test["label"], y_pred_weighted, average="macro")
    print("\n=== BERTweet Weighted ===")
    print(classification_report(df_test["label"], y_pred_weighted, target_names=["Not Racist", "Racist"]))
    weighted_group_results = evaluate_per_group(df_test, y_pred_weighted)
    print_group_results(weighted_group_results, "BERTweet Weighted")

    # Final comparison
    print("\n[7/7] Comparaison finale...")
    print("\n=== Résultats globaux ===")
    summary = pd.DataFrame({
        "Modèle"  : ["SVM + TF-IDF", "BERTweet Baseline", "BERTweet Weighted"],
        "Macro F1": [round(svm_f1_global, 3), round(base_f1_global, 3), round(weighted_f1_global, 3)],
    })
    print(summary.to_string(index=False))

    # Save results as JSON
    results = {
        "svm_f1_global"     : svm_f1_global,
        "base_f1_global"    : base_f1_global,
        "weighted_f1_global": weighted_f1_global,
        "svm_per_group"     : {g: v["f1"] for g, v in svm_group_results.items()},
        "base_per_group"    : {g: v["f1"] for g, v in base_group_results.items()},
        "weighted_per_group": {g: v["f1"] for g, v in weighted_group_results.items()},
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés → {args.output_dir}/results.json")

    save_plots(
        history_base, history_weighted,
        (svm_group_results, base_group_results, weighted_group_results),
        args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Racism detection pipeline")
    parser.add_argument("--data_path",  type=str, required=True,
                        help="Path to hatexplain.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results, checkpoints and plots")
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--patience",   type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_len",    type=int, default=128)
    args = parser.parse_args()
    main(args)
