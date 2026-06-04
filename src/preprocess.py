import json
import os
import re
import pandas as pd

# Define the set of racial targets to identify in the HateXplain dataset
# Note: RACIAL_TARGETS is redefined here so that preprocess.py can be run as
# a standalone script without depending on the rest of the project
RACIAL_TARGETS = {"African", "Asian", "Jewish", "Arab", "Caucasian", "Hispanic", "Indian", "Islam"}


def clean_tweet(text: str) -> str:
    '''
    Clean the input tweet text by replacing user mentions, URLs, and line breaks with standardized tokens

    Args:
        text (str): The original tweet text to clean
    Returns:
        str: The cleaned tweet text with standardized tokens
    '''
    text = str(text)
    text = re.sub(r"<user>", "@USER", text)
    text = re.sub(r"<url>", "HTTPURL", text)
    text = re.sub(r"http\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\[linebreak\]", " ", text)
    text = text.strip()
    return text


def load_hatexplain(path: str) -> pd.DataFrame:
    '''
    Load and preprocess the HateXplain dataset from the specified JSON file path, extracting relevant information 
    and filtering for racist and normal posts

    Labelling logic:
        - label = 1 (racist hate speech) if the majority annotator label is "hatespeech" AND at least one
        racial target is present
        - label = 0 (normal) if the majority label is "normal"
        - All other posts (offensive but not racial, or undecided) are dropped

    Filtering: only posts targetting exactly one racial group are kept to avoid ambiguity during pre-group fairness evaluation

    Args:
        path (str): The file path to the HateXplain JSON dataset
    Returns:
        pd.DataFrame: A DataFrame containing the processed HateXplain data with columns for id, text, label, and targets
    '''
    with open(path) as f:
        data = json.load(f)

    rows = []
    for post_id, content in data.items():
        text = " ".join(content["post_tokens"])
        labels = [a["label"] for a in content["annotators"]]
        targets = [a["target"] for a in content["annotators"]]
        majority_label = max(set(labels), key=labels.count)
        all_targets = {t for sublist in targets for t in sublist}
        is_racist = int(majority_label == "hatespeech" and bool(all_targets & RACIAL_TARGETS))
        rows.append({
            "id": post_id,
            "text": clean_tweet(text),
            "majority_label": majority_label,
            "targets": list(all_targets),
            "racist": is_racist,
        })

    df = pd.DataFrame(rows)
    df = df[(df["racist"] == 1) | (df["majority_label"] == "normal")].copy()
    df["label"] = df["racist"]

    df = df[df["targets"].apply(lambda t: len(set(t) & RACIAL_TARGETS) == 1)].reset_index(drop=True)
    return df[["id", "text", "label", "targets"]]

if __name__ == "__main__":
    # Run as a standalone script to generate hatexplain.csv from the raw JSON
    # Skips processing if the output file already exists
    output_path = "../data/final_datasets/hatexplain.csv"
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}, skipping.")
    else:
        os.makedirs("../data/final_datasets", exist_ok=True)
        df = load_hatexplain("../data/HateXplain.json")
        df.to_csv(output_path, index=False)
        print(df["label"].value_counts())
