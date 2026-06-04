import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from config import GROUPS

def evaluate_per_group(df_eval, y_pred):
    '''
    Evaluate F1 score for each group in the dataset, considering only samples that belong to that group (positive samples)
    and negative samples. 

    Args:
        df_eval (pd.DataFrame): DataFrame containing the evaluation data with columns "targets" and "label"
        y_pred (list or np.array): Predicted labels for the evaluation data

    Returns:
        dict: A dictionary where keys are group names and values are dictionaries with "f1"
    '''
    results = {}
    y_pred = np.array(y_pred)
    for group in GROUPS:
        mask_pos = df_eval["targets"].apply(lambda t: group in t) & (df_eval["label"] == 1)
        mask_neg = df_eval["label"] == 0
        mask = mask_pos | mask_neg
        if mask_pos.sum() < 1:
            continue
        results[group] = {
            "f1": f1_score(df_eval[mask]["label"].values, y_pred[mask], pos_label=1, zero_division=0),
            "n_pos": int(mask_pos.sum()),
        }
    return results

def compute_sample_weights(df_train):
    '''
    Compute sample weights based on the frequency of each group in the training data.
    The weight for each group is calculated as the ratio of the maximum group count to the count
    of that group, ensuring that groups with fewer samples receive higher weights.

    Args:
        df_train (pd.DataFrame): DataFrame containing the training data with columns "targets" and "label"

    Returns:
        np.array: An array of sample weights corresponding to each row in df_train
    '''
    group_counts = {
        g: max(1, df_train["targets"].apply(lambda t: g in t).sum())
        for g in GROUPS
    }

    max_count = max(group_counts.values())
    group_weights = {g: max_count / c for g, c in group_counts.items()}

    print("\nGroup weights (count-based):")
    for g, w in sorted(group_weights.items(), key=lambda x: -x[1]):
        print(f"  {g:<12} count={group_counts[g]:4d}  weight={w:.2f}")

    def get_weight(row):
        if row["label"] == 0:
            return 1.0
        targets = [t for t in row["targets"] if t in group_weights]
        return max((group_weights[t] for t in targets), default=1.0)

    return df_train.apply(get_weight, axis=1).values

def print_group_results(results, model_name=""):
    '''
    Print the F1 scores for each group in a sorted manner

    Args:
        results (dict): A dictionary where keys are group names and values are dictionaries with "f1" and "n_pos"
        model_name (str): Optional name of the model for display purposes
    
    Returns:
        None
    '''
    print(f"\nPer-group F1 ({model_name})")
    rows = [
        {"Group": g, "F1": round(v["f1"], 3), "N positive": v["n_pos"]}
        for g, v in sorted(results.items(), key=lambda x: -x[1]["f1"])
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def get_weight(row):
    '''
    Get the weight for a given row based on the group weights

    Args:
        row (pd.Series): A row from the DataFrame containing "targets" and "label"
    
    Returns:
        float: The weight for the given row
    '''
    if row["label"] == 0:
        return 1.0
    targets = [t for t in row["targets"] if t in group_weights]
    return max((group_weights[t] for t in targets), default=1.0)


def compute_disparity_weights(df_train):
    '''
    Compute disparity-aware sample weights based on the frequency of each group in the training data,
    considering both positive and negative samples. The weight for each group is calculated as the
    ratio of the maximum count of positive samples to the count of positive samples for that group
    multiplied by the ratio of negative samples to positive samples for that group, ensuring that
    groups with fewer positive samples receive higher weights, and groups with a higher imbalance between
    positive and negative samples also receive higher weights

    Args:
        df_train (pd.DataFrame): DataFrame containing the training data with columns "targets" and "label"
    
    Returns:
        np.array: An array of sample weights corresponding to each row in df_train
    '''
    hate_counts = {
        g: max(1, ((df_train["label"] == 1) & df_train["targets"].apply(lambda t: g in t)).sum())
        for g in GROUPS
    }
    normal_counts = {
        g: max(1, ((df_train["label"] == 0) & df_train["targets"].apply(lambda t: g in t)).sum())
        for g in GROUPS
    }
    max_hate = max(hate_counts.values())

    group_weights = {
        g: (max_hate / hate_counts[g]) * max(1.0, normal_counts[g] / hate_counts[g])
        for g in GROUPS
    }

    print("\nGroup weights (disparity-aware):")
    for g, w in sorted(group_weights.items(), key=lambda x: -x[1]):
        print(f"  {g:<12} hate={hate_counts[g]:4d}  normal={normal_counts[g]:4d}  weight={w:.2f}")

    def get_weight(row):
        if row["label"] == 0:
            return 1.0
        targets = [t for t in row["targets"] if t in group_weights]
        return max((group_weights[t] for t in targets), default=1.0)

    return df_train.apply(get_weight, axis=1).values