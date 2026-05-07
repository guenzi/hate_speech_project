# ── Plots.py (nouveaux plots) ──────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os
from math import pi
from sklearn.metrics import confusion_matrix
from config import GROUPS

COLORS = {
    "SVM": "steelblue",
    "BERTweet_baseline": "tomato",
    "BERTweet_weighted": "seagreen",
    "BERTweet_weighted_aug": "darkorchid",
    "BERTweet_disparity": "darkorange",
}


# ── 1. Radar / Spider chart ────────────────────────────────────────────────
def save_radar_chart(all_results, output_dir):
    """
    Un radar par modèle, tous superposés.
    Idéal pour repérer d'un coup d'œil les groupes systématiquement faibles.
    """
    os.makedirs(output_dir, exist_ok=True)
    N = len(GROUPS)
    angles = [n / N * 2 * pi for n in range(N)] + [0]  # fermer le polygone

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(GROUPS, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=7, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    for name, res in all_results.items():
        values = [res.get(g, {}).get("f1", 0.0) for g in GROUPS] + [
            res.get(GROUPS[0], {}).get("f1", 0.0)
        ]
        color = COLORS.get(name, "gray")
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Per-Group F1 — Radar", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── 2. Fairness Gap (équité inter-groupes) ────────────────────────────────
def save_fairness_gap(all_results, output_dir):
    """
    Pour chaque modèle : moyenne, min, max et std du F1 entre groupes.
    Un modèle "équitable" a une faible std et un min élevé.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names, means, stds, mins, maxs = [], [], [], [], []
    for name, res in all_results.items():
        scores = [res.get(g, {}).get("f1", 0.0) for g in GROUPS if g in res]
        if not scores:
            continue
        model_names.append(name)
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        mins.append(np.min(scores))
        maxs.append(np.max(scores))

    x = np.arange(len(model_names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gauche : mean F1 + erreur = std
    colors = [COLORS.get(n, "gray") for n in model_names]
    axes[0].bar(x, means, color=colors, alpha=0.8, yerr=stds, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=20, ha="right")
    axes[0].set_ylabel("Mean F1 across groups")
    axes[0].set_title("Mean F1 ± Std Dev (fairness proxy)")
    axes[0].set_ylim(0, 1.1)

    # Droite : spread min-max (worst-case group)
    spreads = [mx - mn for mx, mn in zip(maxs, mins)]
    axes[1].bar(x, spreads, color=colors, alpha=0.8)
    for i, (mn, mx) in enumerate(zip(mins, maxs)):
        axes[1].text(i, spreads[i] + 0.01, f"{mn:.2f}–{mx:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=20, ha="right")
    axes[1].set_ylabel("Max F1 − Min F1")
    axes[1].set_title("F1 Spread (max − min) per Model")
    axes[1].set_ylim(0, 1.1)

    plt.suptitle("Fairness Analysis — Inter-group Disparity", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_gap.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── 3. FPR vs FNR par groupe ──────────────────────────────────────────────
def save_fpr_fnr_per_group(df_eval, predictions_dict, output_dir):
    """
    predictions_dict = {"model_name": y_pred_array, ...}
    Scatter plot : FPR (axe x) vs FNR (axe y) pour chaque (groupe, modèle).
    Un bon modèle est en bas à gauche.
    Révèle si un modèle rate plus les minorités (FNR élevé) ou les sur-signale (FPR élevé).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, y_pred in predictions_dict.items():
        y_pred = np.array(y_pred)
        color = COLORS.get(name, "gray")
        fprs, fnrs, labels = [], [], []

        for group in GROUPS:
            mask_pos = df_eval["targets"].apply(lambda t: group in t) & (df_eval["label"] == 1)
            mask_neg = df_eval["label"] == 0
            mask = mask_pos | mask_neg
            if mask_pos.sum() < 1 or mask_neg.sum() < 1:
                continue

            y_true_g = df_eval[mask]["label"].values
            y_pred_g = y_pred[mask]
            tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fprs.append(fpr)
            fnrs.append(fnr)
            labels.append(group)

        ax.scatter(fprs, fnrs, color=color, s=80, alpha=0.85, label=name, zorder=3)
        for fpr, fnr, lbl in zip(fprs, fnrs, labels):
            ax.annotate(lbl, (fpr, fnr), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=color, alpha=0.9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("False Positive Rate (over-signaling hate)")
    ax.set_ylabel("False Negative Rate (missing hate)")
    ax.set_title("FPR vs FNR per Group — Error Profile")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fpr_fnr_scatter.png"), dpi=150)
    plt.close()


# ── 4. Delta F1 vs Baseline ───────────────────────────────────────────────
def save_delta_f1(all_results, baseline_name, output_dir):
    """
    Différence de F1 par rapport au modèle baseline, par groupe.
    Barres rouges = régression, vertes = amélioration.
    """
    os.makedirs(output_dir, exist_ok=True)
    baseline = all_results[baseline_name]
    other_models = {k: v for k, v in all_results.items() if k != baseline_name}
    n = len(other_models)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, other_models.items()):
        deltas = [
            res.get(g, {}).get("f1", 0.0) - baseline.get(g, {}).get("f1", 0.0)
            for g in GROUPS
        ]
        bar_colors = ["seagreen" if d >= 0 else "tomato" for d in deltas]
        ax.barh(GROUPS, deltas, color=bar_colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{name}\nvs {baseline_name}", fontsize=9)
        ax.set_xlabel("ΔF1")
        for i, d in enumerate(deltas):
            ax.text(d + (0.005 if d >= 0 else -0.005), i,
                    f"{d:+.3f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=7)

    fig.suptitle(f"F1 Delta vs {baseline_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_f1_vs_baseline.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── 5. Heatmap de co-occurrence des groupes ───────────────────────────────
def save_group_cooccurrence(df, output_dir):
    """
    Combien de tweets mentionnent deux groupes simultanément ?
    Révèle la structure du dataset (groupes souvent co-ciblés vs isolés).
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(GROUPS)
    matrix = np.zeros((n, n), dtype=int)

    for i, g1 in enumerate(GROUPS):
        for j, g2 in enumerate(GROUPS):
            mask = df["targets"].apply(lambda t: g1 in t and g2 in t)
            matrix[i, j] = mask.sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Number of tweets")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(GROUPS, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(GROUPS, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Group Co-occurrence in Dataset\n(# tweets mentioning both groups)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_cooccurrence.png"), dpi=150)
    plt.close()


# ── 6. Bias Amplification ─────────────────────────────────────────────────
def save_bias_amplification(df_eval, predictions_dict, output_dir):
    """
    Pour chaque groupe : ratio (% prédit positif) / (% réellement positif).
    > 1 = sur-signalement, < 1 = sous-signalement.
    Détecte un biais systématique du modèle envers certains groupes.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(GROUPS))
    width = 0.8 / len(predictions_dict)

    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        y_pred = np.array(y_pred)
        ratios = []
        for group in GROUPS:
            mask = df_eval["targets"].apply(lambda t: group in t)
            if mask.sum() == 0:
                ratios.append(1.0)
                continue
            actual_rate = df_eval[mask]["label"].mean()
            pred_rate = y_pred[mask].mean()
            ratio = pred_rate / actual_rate if actual_rate > 0 else 1.0
            ratios.append(ratio)

        offset = (i - len(predictions_dict) / 2 + 0.5) * width
        bars = ax.bar(x + offset, ratios, width, label=name,
                      color=COLORS.get(name, f"C{i}"), alpha=0.85)

    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Perfect parity")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_ylabel("Predicted rate / Actual rate")
    ax.set_title("Bias Amplification per Group\n(>1 = over-prediction, <1 = under-prediction)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_amplification.png"), dpi=150)
    plt.close()


# ── 7. Distribution des sample weights ───────────────────────────────────
def save_weight_distribution(df_train, weights, title, output_dir):
    """
    weights = résultat de compute_sample_weights() ou compute_disparity_weights().
    Violin plot par groupe pour voir si les poids sont cohérents.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_per_group = []
    labels = []

    for group in GROUPS:
        mask = (df_train["label"] == 1) & df_train["targets"].apply(lambda t: group in t)
        if mask.sum() == 0:
            continue
        data_per_group.append(weights[mask])
        labels.append(f"{group}\n(n={mask.sum()})")

    fig, ax = plt.subplots(figsize=(13, 5))
    parts = ax.violinplot(data_per_group, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Sample weight")
    ax.set_title(f"Sample Weight Distribution per Group — {title}")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + "_weight_dist.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    
def save_training_curves(history, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    color = COLORS.get(model_name, "steelblue")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], color=color, marker="o")
    axes[0].set_title(f"{model_name} — Training Loss")
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

def save_dataset_distribution(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    counts_hate = [((df["label"] == 1) & df["targets"].apply(lambda t: g in t)).sum() for g in GROUPS]
    counts_normal = [((df["label"] == 0) & df["targets"].apply(lambda t: g in t)).sum() for g in GROUPS]

    x = np.arange(len(GROUPS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, counts_hate, width, label="Hate (label=1)", color="tomato", alpha=0.85)
    ax.bar(x + width / 2, counts_normal, width, label="Not Hate (label=0)", color="steelblue", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_ylabel("Number of tweets")
    ax.set_title("Hate vs Not-Hate Distribution per Ethnic Group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_distribution.png"), dpi=150)
    plt.close()

def save_group_comparison(all_results, output_dir):
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
    ax.set_title("Per-Group F1 Score — Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_f1_comparison.png"), dpi=150)
    plt.close()