import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from augment import augment_training_data, augment_training_data_disparity
from config import MODEL_NAME, SEED, get_device
from dataset import load_data, make_loaders, split_data
from evaluate import compute_disparity_weights, compute_sample_weights, evaluate_per_group
from model import predict, save_checkpoint, train_bertweet, load_checkpoint
from plots import (
    save_dataset_distribution, save_group_comparison, save_training_curves,
    save_radar_chart, save_fairness_gap, save_fpr_fnr_per_group,
    save_delta_f1, save_group_cooccurrence, save_bias_amplification,
    save_weight_distribution,
    save_macro_f1_per_epoch,
)

warnings.filterwarnings("ignore")


def setup_dirs(base_dir):
    paths = {
        "base": base_dir,
        "plots": os.path.join(base_dir, "plots"),
        "json": os.path.join(base_dir, "json"),
        "csv": os.path.join(base_dir, "csv"),
        "ckpt": os.path.join(base_dir, "checkpoints")
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def run_preprocess(args, paths):
    print("\n=== [STEP] PREPROCESS ===")
    df = load_data(args.data_path)
    df_train, df_val, df_test = split_data(df)

    df_train.to_csv(os.path.join(paths["csv"], "train.csv"), index=False)
    df_val.to_csv(os.path.join(paths["csv"], "val.csv"), index=False)
    df_test.to_csv(os.path.join(paths["csv"], "test.csv"), index=False)

    save_dataset_distribution(df, paths["plots"])


def run_train(args, paths, device):
    print("\n=== [STEP] TRAINING ===")
    df_train = load_data(os.path.join(paths["csv"], "train.csv"))
    df_val = load_data(os.path.join(paths["csv"], "val.csv"))
    df_test = load_data(os.path.join(paths["csv"], "test.csv"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    selected_methods = tuple(m.strip() for m in args.aug_methods.split(",") if m.strip())
    histories = {}

    # --- [1/5] SVM Baseline (TF-IDF) ---
    print("\n[1/5] Training SVM Baseline...")
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC(class_weight='balanced'))
    ])
    svm_pipeline.fit(df_train['text'], df_train['label'])
    joblib.dump(svm_pipeline, os.path.join(paths["ckpt"], "svm.joblib"))

    # --- [2/5] BERTweet Baseline ---
    print("\n[2/5] Training BERTweet Baseline...")
    train_loader, val_loader, _ = make_loaders(df_train, df_val, df_test, tokenizer, args.max_len, args.batch_size)
    model_base, history = train_bertweet(train_loader, val_loader, device, args.epochs, args.lr, patience=args.patience)
    save_checkpoint(model_base, os.path.join(paths["ckpt"], "bertweet_baseline.pt"))
    histories["bertweet_baseline"] = history

    # --- [3/5] BERTweet Weighted ---
    print("\n[3/5] Training BERTweet Weighted (No Aug)...")
    sample_weights = compute_sample_weights(df_train)
    train_loader_w, _, _ = make_loaders(df_train, df_val, df_test, tokenizer, args.max_len, args.batch_size, sample_weights)
    model_w, history = train_bertweet(train_loader_w, val_loader, device, args.epochs, args.lr, use_sample_weights=True, patience=args.patience)
    save_checkpoint(model_w, os.path.join(paths["ckpt"], "bertweet_weighted.pt"))
    histories["bertweet_weighted"] = history

    # --- [4/5] BERTweet Weighted + Aug ---
    print("\n[4/5] Training BERTweet Weighted + Aug...")
    df_train_aug = augment_training_data(df_train, aug_factor=args.aug_factor, seed=SEED, methods=selected_methods)
    sample_weights_aug = compute_sample_weights(df_train_aug)
    train_loader_aug, _, _ = make_loaders(df_train_aug, df_val, df_test, tokenizer, args.max_len, args.batch_size, sample_weights_aug)
    model_aug, history = train_bertweet(train_loader_aug, val_loader, device, args.epochs, args.lr, use_sample_weights=True, patience=args.patience)
    save_checkpoint(model_aug, os.path.join(paths["ckpt"], "bertweet_weighted_aug.pt"))
    histories["bertweet_weighted_aug"] = history

    # --- [5/5] BERTweet + Disparity ---
    print("\n[5/5] Training BERTweet + Disparity...")
    df_train_disp = augment_training_data_disparity(df_train, aug_factor=args.aug_factor, seed=SEED, methods=selected_methods)
    sample_weights_disp = compute_disparity_weights(df_train_disp)
    train_loader_disp, _, _ = make_loaders(df_train_disp, df_val, df_test, tokenizer, args.max_len, args.batch_size, sample_weights_disp)
    model_disp, history = train_bertweet(train_loader_disp, val_loader, device, args.epochs, args.lr, use_sample_weights=True, patience=args.patience)
    save_checkpoint(model_disp, os.path.join(paths["ckpt"], "bertweet_disparity.pt"))
    histories["bertweet_disparity"] = history

    with open(os.path.join(paths["json"], "histories.json"), "w") as f:
        json.dump(histories, f, indent=2)

def run_eval(args, paths, device):
    print("\n=== [STEP] EVALUATION ===")
    df_test = load_data(os.path.join(paths["csv"], "test.csv"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    _, _, test_loader = make_loaders(df_test, df_test, df_test, tokenizer, args.max_len, args.batch_size)

    all_group_results = {}
    all_predictions = {}

    for ckpt in os.listdir(paths["ckpt"]):
        if ckpt.endswith(".joblib"):
            name = ckpt.replace(".joblib", "")
            model = joblib.load(os.path.join(paths["ckpt"], ckpt))
            # Le SVM prédit directement sur le texte brut
            y_pred = model.predict(df_test["text"].values)

        elif ckpt.endswith(".pt"):
            name = ckpt.replace(".pt", "")
            model = load_checkpoint(os.path.join(paths["ckpt"], ckpt), device)
            y_pred = predict(model, test_loader, device)

        else:
            continue

        all_group_results[name] = evaluate_per_group(df_test, y_pred)
        all_predictions[name] = np.array(y_pred).tolist()

    with open(os.path.join(paths["json"], "results.json"), "w") as f:
        json.dump(all_group_results, f, indent=2)
    with open(os.path.join(paths["json"], "predictions.json"), "w") as f:
        json.dump(all_predictions, f)

def run_plot(args, paths):
    with open(os.path.join(paths["json"], "results.json"), "r") as f:
        all_group_results = json.load(f)
    with open(os.path.join(paths["json"], "predictions.json"), "r") as f:
        predictions_dict = {k: np.array(v) for k, v in json.load(f).items()}
    with open(os.path.join(paths["json"], "histories.json"), "r") as f:  # ← nouveau
        histories = json.load(f)
    
    df_test = load_data(os.path.join(paths["csv"], "test.csv"))
    df_train = load_data(os.path.join(paths["csv"], "train.csv"))

    # Plots existants
    save_group_comparison(all_group_results, paths["plots"])

    # Nouveaux plots
    save_radar_chart(all_group_results, paths["plots"])
    save_fairness_gap(all_group_results, paths["plots"])
    save_delta_f1(all_group_results, baseline_name="bertweet_baseline", output_dir=paths["plots"])
    save_group_cooccurrence(df_train, paths["plots"])
    save_fpr_fnr_per_group(df_test, predictions_dict, paths["plots"])
    save_bias_amplification(df_test, predictions_dict, paths["plots"])

    # Weight distributions
    weights_count = compute_sample_weights(df_train)
    weights_disp = compute_disparity_weights(df_train)
    save_weight_distribution(df_train, weights_count, "Count-based", paths["plots"])
    save_weight_distribution(df_train, weights_disp, "Disparity-aware", paths["plots"])
    save_macro_f1_per_epoch(histories, paths["plots"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["preprocess", "train", "eval", "plot", "all"], default="all")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--aug_factor", type=int, default=2)
    parser.add_argument("--aug_methods", type=str, default="synonym,delete,swap,punct,char,combo")

    args = parser.parse_args()
    device = get_device()
    paths = setup_dirs(args.output_dir)

    if args.mode in ["preprocess", "all"]: run_preprocess(args, paths)
    if args.mode in ["train", "all"]: run_train(args, paths, device)
    if args.mode in ["eval", "all"]: run_eval(args, paths, device)
    if args.mode in ["plot", "all"]: run_plot(args, paths)