import os

import matplotlib.pyplot as plt
import numpy as np

from config import GROUPS

COLORS = {
    "SVM": "steelblue",
    "BERTweet Baseline": "tomato",
    "BERTweet Weighted": "seagreen",
    "BERTweet Weighted + Aug": "darkorchid",
}


def save_training_curves(history, model_name, output_dir):
    """Sauvegarde les courbes loss/F1 pour un modèle donné."""
    os.makedirs(output_dir, exist_ok=True)
    color = COLORS.get(model_name, "steelblue")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], color=color, marker="o")
    axes[0].set_title(f"{model_name} — Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(history["val_f1"], color=color, marker="o")
    axes[1].set_title(f"{model_name} — Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")

    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_").replace("+", "plus") + "_curves.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"Courbes sauvegardées → {filename}")


def save_dataset_distribution(df, output_dir):
    """Bar chart montrant la disparité hate vs not hate par groupe ethnique."""
    os.makedirs(output_dir, exist_ok=True)

    counts_hate = [((df["label"] == 1) & df["targets_parsed"].apply(lambda t: g in t)).sum() for g in GROUPS]
    counts_normal = [((df["label"] == 0) & df["targets_parsed"].apply(lambda t: g in t)).sum() for g in GROUPS]

    x = np.arange(len(GROUPS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, counts_hate, width, label="Hate (label=1)", color="tomato", alpha=0.85)
    ax.bar(x + width / 2, counts_normal, width, label="Not Hate (label=0)", color="steelblue", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_ylabel("Nombre de tweets")
    ax.set_title("Distribution hate vs not hate par groupe ethnique")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_distribution.png"), dpi=150)
    plt.close()
    print("Distribution sauvegardée → dataset_distribution.png")


def save_group_comparison(all_results, output_dir):
    """
    all_results : dict { model_name: group_results_dict }
    Sauvegarde un bar chart comparant tous les modèles par groupe ethnique.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(all_results.keys())
    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(len(GROUPS))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(model_names):
        res = all_results[name]
        f1_scores = [res.get(g, {}).get("f1", 0.0) for g in GROUPS]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, label=name,
               color=COLORS.get(name, f"C{i}"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 par groupe ethnique — comparaison des modèles")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_f1_comparison.png"), dpi=150)
    plt.close()
    print("Comparaison sauvegardée → group_f1_comparison.png")
