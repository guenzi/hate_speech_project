"""
Racism Detection Pipeline — Cluster version
Usage:
    python3 main.py --data_path /path/to/hatexplain.csv --output_dir /path/to/results/
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from augment import augment_training_data, augment_training_data_disparity
from config import MODEL_NAME, SEED, get_device
from dataset import load_data, make_loaders, split_data
from evaluate import compute_disparity_weights, compute_sample_weights, evaluate_per_group, print_group_results
from model import predict, save_checkpoint, train_bertweet
from plots import save_dataset_distribution, save_group_comparison, save_training_curves

warnings.filterwarnings("ignore")


def main(args):
    device = get_device()
    print(f"Device : {device}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Run ID : {run_id}  →  résultats dans {args.output_dir}")

    # Chargement et split des données
    print("\n[1] Chargement des données...")
    df = load_data(args.data_path)
    print(f"Total : {len(df)} | Racist: {(df['label']==1).sum()} | Normal: {(df['label']==0).sum()}")
    df_train, df_val, df_test = split_data(df)
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    save_dataset_distribution(df, args.output_dir)

    all_group_results = {}
    all_f1_global = {}

    # SVM baseline
    print("\n[2] SVM + TF-IDF...")
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=SEED)),
    ])
    svm_pipe.fit(df_train["text"], df_train["label"])
    y_pred_svm = svm_pipe.predict(df_test["text"])
    all_f1_global["SVM"] = f1_score(df_test["label"], y_pred_svm, average="macro")
    print(classification_report(df_test["label"], y_pred_svm, target_names=["Not Racist", "Racist"]))
    all_group_results["SVM"] = evaluate_per_group(df_test, y_pred_svm)
    print_group_results(all_group_results["SVM"], "SVM")

    # Tokenizer BERTweet
    print("\n[3] Chargement tokenizer BERTweet...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    train_loader, val_loader, test_loader = make_loaders(
        df_train, df_val, df_test, tokenizer, args.max_len, args.batch_size
    )

    # BERTweet baseline
    print("\n[4] BERTweet Baseline...")
    model_base, history_base = train_bertweet(
        train_loader, val_loader, device, args.epochs, args.lr,
        use_sample_weights=False, patience=args.patience,
    )
    save_checkpoint(model_base, os.path.join(args.output_dir, "checkpoints", "bertweet_baseline.pt"))
    save_training_curves(history_base, "BERTweet Baseline", args.output_dir)

    y_pred_base = predict(model_base, test_loader, device)
    all_f1_global["BERTweet Baseline"] = f1_score(df_test["label"], y_pred_base, average="macro")
    print(classification_report(df_test["label"], y_pred_base, target_names=["Not Racist", "Racist"]))
    all_group_results["BERTweet Baseline"] = evaluate_per_group(df_test, y_pred_base)
    print_group_results(all_group_results["BERTweet Baseline"], "BERTweet Baseline")

    # BERTweet weighted
    print("\n[5] BERTweet Weighted...")
    sample_weights = compute_sample_weights(df_train)
    train_loader_w, _, _ = make_loaders(
        df_train, df_val, df_test, tokenizer, args.max_len, args.batch_size, sample_weights
    )
    model_weighted, history_weighted = train_bertweet(
        train_loader_w, val_loader, device, args.epochs, args.lr,
        use_sample_weights=True, patience=args.patience,
    )
    save_checkpoint(model_weighted, os.path.join(args.output_dir, "checkpoints", "bertweet_weighted.pt"))
    save_training_curves(history_weighted, "BERTweet Weighted", args.output_dir)

    y_pred_weighted = predict(model_weighted, test_loader, device)
    all_f1_global["BERTweet Weighted"] = f1_score(df_test["label"], y_pred_weighted, average="macro")
    print(classification_report(df_test["label"], y_pred_weighted, target_names=["Not Racist", "Racist"]))
    all_group_results["BERTweet Weighted"] = evaluate_per_group(df_test, y_pred_weighted)
    print_group_results(all_group_results["BERTweet Weighted"], "BERTweet Weighted")

    # BERTweet weighted + augmentation
    print("\n[6] BERTweet Weighted + Augmentation...")
    aug_report_path = os.path.join(args.output_dir, "augmentation_report.json")
    selected_methods = tuple(m.strip() for m in args.aug_methods.split(",") if m.strip())
    df_train_aug = augment_training_data(
        df_train,
        aug_factor=args.aug_factor,
        min_group_samples=args.aug_min_group_samples,
        seed=SEED,
        methods=selected_methods,
        save_report_path=aug_report_path,
    )
    df_train_aug.to_csv(os.path.join(args.output_dir, "train_augmented.csv"), index=False)

    sample_weights_aug = compute_sample_weights(df_train_aug)
    train_loader_aug, _, _ = make_loaders(
        df_train_aug, df_val, df_test, tokenizer, args.max_len, args.batch_size, sample_weights_aug
    )
    model_aug, history_aug = train_bertweet(
        train_loader_aug, val_loader, device, args.epochs, args.lr,
        use_sample_weights=True, patience=args.patience,
    )
    save_checkpoint(model_aug, os.path.join(args.output_dir, "checkpoints", "bertweet_weighted_aug.pt"))
    save_training_curves(history_aug, "BERTweet Weighted + Aug", args.output_dir)

    y_pred_aug = predict(model_aug, test_loader, device)
    all_f1_global["BERTweet Weighted + Aug"] = f1_score(df_test["label"], y_pred_aug, average="macro")
    print(classification_report(df_test["label"], y_pred_aug, target_names=["Not Racist", "Racist"]))
    all_group_results["BERTweet Weighted + Aug"] = evaluate_per_group(df_test, y_pred_aug)
    print_group_results(all_group_results["BERTweet Weighted + Aug"], "BERTweet Weighted + Aug")

    # BERTweet + Disparity (disparity-aware weights + disparity-aware augmentation)
    print("\n[7] BERTweet + Disparity...")
    disp_aug_report_path = os.path.join(args.output_dir, "augmentation_disparity_report.json")
    df_train_disp = augment_training_data_disparity(
        df_train,
        aug_factor=args.aug_factor,
        disparity_threshold=args.aug_disparity_threshold,
        seed=SEED,
        methods=selected_methods,
        save_report_path=disp_aug_report_path,
    )
    df_train_disp.to_csv(os.path.join(args.output_dir, "train_disparity_augmented.csv"), index=False)

    sample_weights_disp = compute_disparity_weights(df_train_disp)
    train_loader_disp, _, _ = make_loaders(
        df_train_disp, df_val, df_test, tokenizer, args.max_len, args.batch_size, sample_weights_disp
    )
    model_disp, history_disp = train_bertweet(
        train_loader_disp, val_loader, device, args.epochs, args.lr,
        use_sample_weights=True, patience=args.patience,
    )
    save_checkpoint(model_disp, os.path.join(args.output_dir, "checkpoints", "bertweet_disparity.pt"))
    save_training_curves(history_disp, "BERTweet + Disparity", args.output_dir)

    y_pred_disp = predict(model_disp, test_loader, device)
    all_f1_global["BERTweet + Disparity"] = f1_score(df_test["label"], y_pred_disp, average="macro")
    print(classification_report(df_test["label"], y_pred_disp, target_names=["Not Racist", "Racist"]))
    all_group_results["BERTweet + Disparity"] = evaluate_per_group(df_test, y_pred_disp)
    print_group_results(all_group_results["BERTweet + Disparity"], "BERTweet + Disparity")

    # Résultats globaux
    print("\n=== Résultats globaux ===")
    summary = pd.DataFrame({
        "Modèle": list(all_f1_global.keys()),
        "Macro F1": [round(v, 3) for v in all_f1_global.values()],
    })
    print(summary.to_string(index=False))

    results = {
        "f1_global": all_f1_global,
        "f1_per_group": {
            model: {g: v["f1"] for g, v in res.items()}
            for model, res in all_group_results.items()
        },
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés → {args.output_dir}/results.json")

    # Plot de comparaison global
    save_group_comparison(all_group_results, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Racism detection pipeline")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--aug_factor", type=int, default=2)
    parser.add_argument("--aug_min_group_samples", type=int, default=200)
    parser.add_argument("--aug_disparity_threshold", type=float, default=1.5)
    parser.add_argument("--aug_methods", type=str, default="synonym,delete,swap,punct,char,combo")
    args = parser.parse_args()
    main(args)
