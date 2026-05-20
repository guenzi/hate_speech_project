# ── Plots.py (style présentation — navy + rouge + fond gris) ──────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Fixé : Import propre pour éviter la NameError !
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import os
from math import pi
from sklearn.metrics import confusion_matrix
from config import GROUPS

# ── Palette & style global ────────────────────────────────────────────────

NAVY   = "#1a2e6e"
RED    = "#cc2233"
SILVER = "#b0b8c9"
BG     = "#F3FFF4"       # fond figure
PANEL  = "#f5f5f5"       # fond axes

COLORS = {
    "SVM":                    NAVY,
    "bertweet_baseline":      "#2c6fad",
    "bertweet_weighted":      "#1a7a4a",
    "bertweet_weighted_aug":  "#7a2d8c",
    "bertweet_disparity":     "#c96a00",
}

# groupes sous-représentés (barres rouges dans image 1)
UNDERREPRESENTED = {"Hispanic", "Asian", "Arab", "Indian"}

# ── rcParams communs ──────────────────────────────────────────────────────

def _apply_style():
    # FACTEUR DE GRANDEUR (Scale) : On augmente la base des polices à 14 (au lieu de 9-10)
    # Matplotlib va automatiquement adapter la taille des titres, labels et légendes.
    base_size = 14
    
    plt.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     PANEL,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": True,
        "axes.edgecolor":     "#999999",
        "axes.grid":          True,
        "grid.color":         "#cccccc",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.8,  # Légèrement plus épais pour les grands plots
        "axes.axisbelow":     True,
        "font.family":        "DejaVu Sans",
        "font.weight":        "bold",
        "font.size":          base_size,       # Échelle globale augmentée
        "axes.titlesize":     base_size + 4,   # Titres plus grands
        "axes.titleweight":   "bold",
        "axes.titlecolor":    NAVY,
        "axes.labelsize":     base_size + 1,   # Axes labels plus grands
        "axes.labelcolor":    "#333333",
        "xtick.labelsize":    base_size - 1,   # Graduations X plus grandes
        "ytick.labelsize":    base_size - 1,   # Graduations Y plus grandes
        "xtick.color":        "#444444",
        "ytick.color":        "#444444",
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#cccccc",
        "legend.fontsize":    base_size - 1,   # Légende plus grande
    })

def _spine_clean(ax):
    """Garde uniquement le bas, supprime les ticks Y."""
    ax.yaxis.set_tick_params(length=0)
    ax.tick_params(axis="y", which="both", left=False)

def _bar_labels(ax, bars, fmt="{:.2f}", offset=0.012, fontsize=12, color=None):
    """Annote chaque barre avec sa valeur (fonts augmentées)."""
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        c = color or bar.get_facecolor()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=c,
        )

def _hbar_labels(ax, bars, fmt="{:+.3f}", offset=0.006, fontsize=11):
    for bar in bars:
        w = bar.get_width()
        if w == 0:
            continue
        ha = "left" if w >= 0 else "right"
        x  = w + offset if w >= 0 else w - offset
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), va="center", ha=ha,
                fontsize=fontsize, fontweight="bold",
                color="seagreen" if w >= 0 else RED)

def _suptitle_box(fig, text, fontsize=18):
    """Titre principal dans un rectangle navy."""
    fig.text(0.5, 1.02, text,
             ha="center", va="bottom",
             fontsize=fontsize, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=NAVY, edgecolor="none"))

def _title_box(ax, text, fontsize=None):
    """Titre d'axe dans un encadré navy arrondi, texte blanc — même style que _suptitle_box."""
    fs = fontsize or plt.rcParams.get("axes.titlesize", 16)
    ax.set_title(
        text,
        fontsize=fs,
        fontweight="bold",
        color="white",
        pad=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=NAVY, edgecolor="none"),
    )


# ── 0. Macro F1 hero card (inspiré image 1) ──────────────────────────────
def save_macro_hero(global_f1, per_group_f1, model_name, output_dir):
    """ Grande carte : macro F1 en haut + barres par groupe (rouge si < 0.6). """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    groups = list(per_group_f1.keys())
    scores = list(per_group_f1.values())
    bar_colors = [RED if s < 0.6 else NAVY for s in scores]

    fig, (ax_hero, ax_bar) = plt.subplots(
        2, 1, figsize=(14, 11),  # Agrandissement proportionnel
        gridspec_kw={"height_ratios": [1, 3]},
    )
    fig.subplots_adjust(hspace=0.4)

    # ── Hero tile ──
    ax_hero.set_axis_off()
    ax_hero.set_facecolor(BG)
    ax_hero.text(0.5, 0.75, "Global Macro F1",
                 ha="center", va="center", transform=ax_hero.transAxes,
                 fontsize=20, fontweight="bold", color=NAVY)
    
    fancy = FancyBboxPatch((0.32, 0.05), 0.36, 0.55,
                           boxstyle="round,pad=0.03",
                           linewidth=0, facecolor=NAVY,
                           transform=ax_hero.transAxes, zorder=2)
    ax_hero.add_patch(fancy)
    ax_hero.text(0.5, 0.32, f"{global_f1:.2f}",
                 ha="center", va="center", transform=ax_hero.transAxes,
                 fontsize=48, fontweight="bold", color="white", zorder=3)

    # ── Bar chart ──
    x = np.arange(len(groups))
    bars = ax_bar.bar(x, scores, color=bar_colors, alpha=0.9, width=0.6,
                      zorder=3, edgecolor="white", linewidth=0.8)
    ax_bar.axhline(global_f1, color=NAVY, linestyle="--", linewidth=2.0,
                   label=f"global F1 = {global_f1:.2f}", zorder=4)
    _bar_labels(ax_bar, bars, fmt="{:.2f}", fontsize=12)
    _spine_clean(ax_bar)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(groups, rotation=20, ha="right")
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("F1 Score")
    _title_box(ax_bar, "Per-group F1 scores")

    legend_patches = [
        mpatches.Patch(color=NAVY, label="Well-represented"),
        mpatches.Patch(color=RED,  label="Underrepresented"),
    ]
    ax_bar.legend(handles=legend_patches + ax_bar.get_legend_handles_labels()[0][1:],
                  loc="upper right", fontsize=11)

    ax_bar.text(0.5, -0.22,
                "* Values are illustrative — real results in Validation section",
                ha="center", fontsize=10, color="#777777",
                transform=ax_bar.transAxes, style="italic")

    # dpi=300 assure une qualité cristal d'impression/présentation
    plt.savefig(os.path.join(output_dir, f"macro_hero_{model_name}.png"),
                dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()


# ── 1. Radar / Spider chart ───────────────────────────────────────────────
def save_radar_chart(all_results, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    N = len(GROUPS)
    angles = [n / N * 2 * pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(GROUPS, size=12, fontweight="bold", color=NAVY)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=10, color="grey")
    ax.grid(color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.spines["polar"].set_visible(False)

    for name, res in all_results.items():
        values = [res.get(g, {}).get("f1", 0.0) for g in GROUPS] + \
                 [res.get(GROUPS[0], {}).get("f1", 0.0)]
        color = COLORS.get(name, "gray")
        ax.plot(angles, values, color=color, linewidth=3.0, label=name, zorder=3)
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
    _title_box(ax, "Per-Group F1 — Radar", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_f1.png"),
                dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()


# ── 2. Fairness Gap ───────────────────────────────────────────────────────
def save_fairness_gap(all_results, output_dir):
    _apply_style()
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.patch.set_facecolor(BG)

    colors = [COLORS.get(n, NAVY) for n in model_names]

    bars0 = ax.bar(x, means, color=colors, alpha=0.88, zorder=3,
                   edgecolor="white", linewidth=0.8)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="#333333", capsize=6, linewidth=2.0, zorder=4)
    _bar_labels(ax, bars0, fmt="{:.2f}", fontsize=12)
    _spine_clean(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Mean F1 across groups")
    _title_box(ax, "Mean F1 ± Std Dev\n(fairness proxy)")
    ax.set_ylim(0, 1.15)

    _suptitle_box(fig, "Fairness Analysis — Inter-group Disparity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_gap.png"),
                dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()


# ── 3. FPR vs FNR par groupe ──────────────────────────────────────────────
def save_fpr_fnr_per_group(df_eval, predictions_dict, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(BG)

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
            fprs.append(fpr); fnrs.append(fnr); labels.append(group)

        # Augmentation taille des marqueurs à 140 pour aller avec les grandes polices
        ax.scatter(fprs, fnrs, color=color, s=140, alpha=0.9, label=name,
                   zorder=4, edgecolors="white", linewidths=0.8)
        for fpr, fnr, lbl in zip(fprs, fnrs, labels):
            ax.annotate(lbl, (fpr, fnr), textcoords="offset points",
                        xytext=(8, 5), fontsize=10, color=color,
                        fontweight="bold", alpha=0.9)

    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=1.2)
    ax.axvline(0.5, color="#aaaaaa", linestyle="--", linewidth=1.2)
    
    # Flèche d'origine restaurée et redimensionnée proprement
    ax.add_patch(mpatches.FancyArrowPatch(
        (0.08, 0.08), (0.01, 0.01),
        arrowstyle="-|>", color=NAVY, linewidth=1.8, mutation_scale=16, zorder=5))
    ax.text(0.10, 0.06, "ideal zone", fontsize=10, color=NAVY, style="italic")

    ax.set_xlabel("False Positive Rate  (over-signaling hate)")
    ax.set_ylabel("False Negative Rate  (missing hate)")
    _title_box(ax, "FPR vs FNR per Group — Error Profile")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fpr_fnr_scatter.png"),
                dpi=300, facecolor=BG)
    plt.close()


# ── 4. Delta F1 vs Baseline ───────────────────────────────────────────────
def save_delta_f1(all_results, baseline_name, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    baseline = all_results[baseline_name]
    other_models = {k: v for k, v in all_results.items() if k != baseline_name}
    n = len(other_models)

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), sharey=True)
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, other_models.items()):
        deltas = [
            res.get(g, {}).get("f1", 0.0) - baseline.get(g, {}).get("f1", 0.0)
            for g in GROUPS
        ]
        bar_colors = ["#1a7a4a" if d >= 0 else RED for d in deltas]
        bars = ax.barh(GROUPS, deltas, color=bar_colors, alpha=0.88,
                       edgecolor="white", linewidth=0.8, zorder=3, height=0.55)
        ax.axvline(0, color="#333333", linewidth=1.2, zorder=4)
        _hbar_labels(ax, bars, fontsize=10)
        _title_box(ax, f"{name}\nvs {baseline_name}", fontsize=12)
        ax.set_xlabel("ΔF1")
        _spine_clean(ax)
        ax.spines["left"].set_visible(False)

    _suptitle_box(fig, f"F1 Delta vs {baseline_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_f1_vs_baseline.png"),
                dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()


# ── 5. Heatmap de co-occurrence ───────────────────────────────────────────
def save_group_cooccurrence(df, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    n = len(GROUPS)
    matrix = np.zeros((n, n), dtype=int)
    for i, g1 in enumerate(GROUPS):
        for j, g2 in enumerate(GROUPS):
            matrix[i, j] = df["targets"].apply(lambda t: g1 in t and g2 in t).sum()

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(BG)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("navy_heat", [PANEL, "#2c6fad", NAVY])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Number of tweets", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(GROUPS, rotation=40, ha="right", fontsize=11)
    ax.set_yticklabels(GROUPS, fontsize=11)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            c = "white" if val > matrix.max() * 0.55 else NAVY
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, color=c, fontweight="bold")

    _title_box(ax, "Group Co-occurrence in Dataset\n(# tweets mentioning both groups)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_cooccurrence.png"),
                dpi=300, facecolor=BG)
    plt.close()


# ── 6. Bias Amplification ─────────────────────────────────────────────────
def save_bias_amplification(df_eval, predictions_dict, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor(BG)

    x = np.arange(len(GROUPS))
    width = 0.8 / len(predictions_dict)

    for i, (name, y_pred) in enumerate(predictions_dict.items()):
        y_pred = np.array(y_pred)
        ratios = []
        for group in GROUPS:
            mask = df_eval["targets"].apply(lambda t: group in t)
            if mask.sum() == 0:
                ratios.append(1.0); continue
            actual_rate = df_eval[mask]["label"].mean()
            pred_rate   = y_pred[mask].mean()
            ratios.append(pred_rate / actual_rate if actual_rate > 0 else 1.0)

        offset = (i - len(predictions_dict) / 2 + 0.5) * width
        bar_colors = [RED if r > 1.2 or r < 0.8 else NAVY for r in ratios]
        bars = ax.bar(x + offset, ratios, width, label=name,
                      color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.7, zorder=3)

    ax.axhline(1.0, color="#333333", linewidth=2.0, linestyle="--",
                label="Perfect parity", zorder=4)
    ax.axhspan(0.8, 1.2, alpha=0.08, color="green", zorder=1)
    ax.text(len(GROUPS) - 0.5, 1.22, "±20 % tolerance",
            fontsize=10, color="#555555", style="italic")

    _spine_clean(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_ylabel("Predicted rate / Actual rate")
    _title_box(ax, "Bias Amplification per Group")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_amplification.png"),
                dpi=300, facecolor=BG)
    plt.close()


# ── 7. Training curves ────────────────────────────────────────────────────
def save_training_curves(history, model_name, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    color = COLORS.get(model_name, NAVY)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG)

    for ax, key, ylabel, title in [
        (axes[0], "train_loss", "Loss",       f"{model_name} — Training Loss"),
        (axes[1], "val_f1",    "Macro F1",    f"{model_name} — Validation Macro F1"),
    ]:
        vals = history[key]
        epochs = range(1, len(vals) + 1)
        ax.plot(epochs, vals, color=color, marker="o", linewidth=2.8,
                markersize=7, zorder=3)
        best_idx = int(np.argmin(vals) if "loss" in key else np.argmax(vals))
        ax.scatter([best_idx + 1], [vals[best_idx]], color=RED, s=120,
                   zorder=5, label=f"best: {vals[best_idx]:.3f}")
        ax.axhline(vals[best_idx], color=RED, linestyle=":", linewidth=1.2, alpha=0.6)
        _spine_clean(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        _title_box(ax, title)
        ax.legend(fontsize=10)

    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_").replace("+", "plus") + "_curves.png"
    plt.savefig(os.path.join(output_dir, filename),
                dpi=300, facecolor=BG)
    plt.close()


# ── 8. Dataset distribution (inspiré image 2) ────────────────────────────
def save_dataset_distribution(df, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    counts_hate   = [((df["label"] == 1) & df["targets"].apply(lambda t: g in t)).sum() for g in GROUPS]
    counts_normal = [((df["label"] == 0) & df["targets"].apply(lambda t: g in t)).sum() for g in GROUPS]

    order = np.argsort(counts_hate)
    groups_sorted  = [GROUPS[i] for i in order]
    hate_sorted    = [counts_hate[i]   for i in order]
    normal_sorted  = [counts_normal[i] for i in order]
    bar_colors     = [RED if hate_sorted[i] < 200 else NAVY
                      for i in range(len(groups_sorted))]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    y = np.arange(len(groups_sorted))

    ax.barh(y, hate_sorted,   color=bar_colors, alpha=0.90,
            label="Hate (label=1)", zorder=3, height=0.55, edgecolor="white")
    ax.barh(y, normal_sorted, left=hate_sorted, color=SILVER, alpha=0.70,
            label="Not Hate (label=0)", zorder=3, height=0.55, edgecolor="white")

    ax.axvline(200, color=RED, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(205, len(groups_sorted) - 0.6, "< 200 hate → augmented",
            fontsize=10, color=RED, style="italic")

    for i, (h, n) in enumerate(zip(hate_sorted, normal_sorted)):
        ax.text(h + n + 25, i, str(h + n), va="center",
                fontsize=10, fontweight="bold", color="#333333")

    _spine_clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_yticks(y)
    ax.set_yticklabels(groups_sorted, fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of posts")
    _title_box(ax, "Posts per Ethnic Group  (hate vs. normal)")
    ax.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_distribution.png"),
                dpi=300, facecolor=BG)
    plt.close()


# ── 9. Group F1 comparison ────────────────────────────────────────────────
def save_group_comparison(all_results, output_dir = "plots"):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(all_results.keys())
    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(len(GROUPS))

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(BG)

    for i, name in enumerate(model_names):
        res = all_results[name]
        f1_scores = [res.get(g, {}).get("f1", 0.0) for g in GROUPS]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=name,
                      color=COLORS.get(name, f"C{i}"), alpha=0.88,
                      edgecolor="white", linewidth=0.6, zorder=3)

    _spine_clean(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    _title_box(ax, "Per-Group F1 Score — Model Comparison")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_f1_comparison.png"),
                dpi=300, facecolor=BG)
    plt.close()


# ── 10. Macro F1 per epoch (toutes courbes) ───────────────────────────────
def save_macro_f1_per_epoch(histories, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(BG)

    for name, history in histories.items():
        val_f1 = history["val_f1"]
        epochs = range(1, len(val_f1) + 1)
        color = COLORS.get(name, "gray")
        ax.plot(epochs, val_f1, color=color, marker="o", linewidth=2.8,
                markersize=6, label=name, zorder=3)
        best_epoch = int(np.argmax(val_f1)) + 1
        best_val   = max(val_f1)
        ax.annotate(f"{best_val:.3f}",
                    xy=(best_epoch, best_val),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")

    _spine_clean(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Macro F1")
    _title_box(ax, "Validation Macro F1 per Epoch — All Models")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "macro_f1_per_epoch.png"),
                dpi=300, facecolor=BG)
    plt.close()


# ── 11. Weight distribution ───────────────────────────────────────────────
def save_weight_distribution(df_train, weights, title, output_dir):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    data_per_group, labels = [], []

    for group in GROUPS:
        mask = (df_train["label"] == 1) & df_train["targets"].apply(lambda t: group in t)
        if mask.sum() == 0:
            continue
        data_per_group.append(weights[mask])
        labels.append(f"{group}\n(n={mask.sum()})")

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor(BG)
    parts = ax.violinplot(data_per_group, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(NAVY)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color(RED)
    parts["cmedians"].set_linewidth(2.5)
    parts["cmaxes"].set_color(NAVY)
    parts["cmins"].set_color(NAVY)
    parts["cbars"].set_color(NAVY)

    _spine_clean(ax)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Sample weight")
    _title_box(ax, f"Sample Weight Distribution per Group — {title}")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + "_weight_dist.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300, facecolor=BG)
    plt.close()


# ── 12. Macro F1 vs Per-Group F1 — tous modèles (grille) ─────────────────
def save_macro_vs_group_f1(all_results, output_dir="plots"):
    import math
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(all_results.keys())
    n = len(model_names)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), sharey=True)
    fig.patch.set_facecolor(BG)

    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, name in enumerate(model_names):
        ax = axes_flat[i]
        res       = all_results[name]
        f1_scores = [res.get(g, {}).get("f1", 0.0) for g in GROUPS]
        macro_f1  = res.get("__macro_f1__", np.mean(f1_scores))

        bar_colors = [RED if s < macro_f1 else COLORS.get(name, NAVY)
                      for s in f1_scores]

        x    = np.arange(len(GROUPS))
        bars = ax.bar(x, f1_scores, color=bar_colors, alpha=0.88, width=0.6,
                      edgecolor="white", linewidth=0.7, zorder=3)

        ax.axhline(macro_f1, color=COLORS.get(name, NAVY),
                   linestyle="--", linewidth=2.0,
                   label=f"macro F1 = {macro_f1:.2f}", zorder=4)

        _bar_labels(ax, bars, fmt="{:.2f}", fontsize=10)
        _spine_clean(ax)

        ax.set_xticks(x)
        ax.set_xticklabels(GROUPS, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 1.15)
        _title_box(ax, name, fontsize=13)
        ax.legend(fontsize=10, loc="upper right")

        if i % ncols == 0:
            ax.set_ylabel("F1 Score")

    for j in range(n, nrows * ncols):
        axes_flat[j].set_visible(False)

    _suptitle_box(fig, "Macro F1 vs Per-Group F1 — All Models")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "macro_vs_group_f1.png"),
                dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()