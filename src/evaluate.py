import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import GROUPS


def evaluate_per_group(df_eval, y_pred):
    results = {}
    y_pred = np.array(y_pred)
    for group in GROUPS:
        mask_pos = df_eval["targets_parsed"].apply(lambda t: group in t) & (df_eval["label"] == 1)
        mask_neg = df_eval["label"] == 0
        mask = mask_pos | mask_neg
        if mask_pos.sum() < 1:
            continue
        results[group] = {
            "f1": f1_score(df_eval[mask]["label"].values, y_pred[mask], pos_label=1, zero_division=0),
            "n_pos": int(mask_pos.sum()),
        }
    return results


def print_group_results(results, model_name=""):
    print(f"\nF1 par groupe ({model_name})")
    rows = [
        {"Groupe": g, "F1": round(v["f1"], 3), "N positifs": v["n_pos"]}
        for g, v in sorted(results.items(), key=lambda x: -x[1]["f1"])
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def compute_sample_weights(df_train):
    group_counts = {
        g: max(1, df_train["targets_parsed"].apply(lambda t: g in t).sum())
        for g in GROUPS
    }
    max_count = max(group_counts.values())
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
