import json
import os
import re
import pandas as pd

RACIAL_TARGETS = {"African", "Asian", "Jewish", "Arab", "Caucasian", "Hispanic", "Indian", "Islam"}


def clean_tweet(text: str) -> str:
    text = str(text)
    text = re.sub(r"<user>", "@USER", text)
    text = re.sub(r"<url>", "HTTPURL", text)
    text = re.sub(r"http\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\[linebreak\]", " ", text)
    text = text.strip()
    return text


def load_hatexplain(path: str) -> pd.DataFrame:
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
            "text": text,
            "majority_label": majority_label,
            "targets": list(all_targets),
            "racist": is_racist,
        })

    df = pd.DataFrame(rows)
    df = df[(df["racist"] == 1) | (df["majority_label"] == "normal")].copy()
    df["label"] = df["racist"]
    return df[["id", "text", "label", "targets"]].reset_index(drop=True)


def load_implicit_hate(stg1_path: str, stg2_path: str) -> pd.DataFrame:
    """
    Build implicit hate test set:
      label=1 → implicit_hate posts from stg2 (with sub-category)
      label=0 → not_hate posts from stg1 (same corpus, clean negatives)
    """
    stg1 = pd.read_csv(stg1_path, sep="\t").dropna(subset=["post", "class"])
    stg2 = pd.read_csv(stg2_path, sep="\t").dropna(subset=["post", "implicit_class"])

    negatives = stg1[stg1["class"] == "not_hate"][["post"]].copy()
    negatives["label"] = 0
    negatives["implicit_category"] = "not_hate"

    positives = stg2[["post", "implicit_class"]].copy()
    positives["label"] = 1
    positives = positives.rename(columns={"implicit_class": "implicit_category"})

    df = pd.concat([positives, negatives], ignore_index=True)
    df = df.rename(columns={"post": "text"})
    df["text"] = df["text"].apply(clean_tweet)
    return df[["text", "label", "implicit_category"]].reset_index(drop=True)


def load_cad(train_path: str, test_path: str = None) -> pd.DataFrame:
    """
    CAD: IdentityDirectedAbuse → label=1, Neutral → label=0.
    Note: IdentityDirectedAbuse includes racism but also other identity-based abuse.
    """
    dfs = [pd.read_csv(train_path, sep="\t")]
    if test_path:
        dfs.append(pd.read_csv(test_path, sep="\t"))
    df = pd.concat(dfs, ignore_index=True)

    df = df.dropna(subset=["text", "labels"])
    df["label"] = df["labels"].apply(
        lambda x: 1 if "IdentityDirectedAbuse" in x else (0 if x == "Neutral" else None)
    )
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_tweet)
    return df[["text", "label"]].reset_index(drop=True)


def load_gab(train_path: str, test_path: str = None) -> pd.DataFrame:
    """
    Gab Hate Corpus: hd=1 → hate speech (label=1), hd=0 → label=0.
    """
    dfs = [pd.read_csv(train_path, sep="\t")]
    if test_path:
        dfs.append(pd.read_csv(test_path, sep="\t"))
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["text", "hd"])
    df["label"] = df["hd"].astype(int)
    df["text"] = df["text"].apply(clean_tweet)
    return df[["text", "label"]].reset_index(drop=True)


def preprocess(hatexplain_path: str) -> pd.DataFrame:
    df = load_hatexplain(hatexplain_path)
    df["text"] = df["text"].apply(clean_tweet)
    return df


def save_all(
    hatexplain_path: str,
    stg1_path: str,
    stg2_path: str,
    cad_train_path: str,
    cad_test_path: str,
    gab_train_path: str,
    gab_test_path: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    df_hx = preprocess(hatexplain_path)
    df_hx.to_csv(os.path.join(output_dir, "hatexplain.csv"), index=False)
    print(f"hatexplain.csv        — {df_hx['label'].value_counts().to_dict()}")

    df_impl = load_implicit_hate(stg1_path, stg2_path)
    df_impl.to_csv(os.path.join(output_dir, "implicit_hate_test.csv"), index=False)
    print(f"implicit_hate_test.csv — {df_impl['label'].value_counts().to_dict()}")

    df_cad = load_cad(cad_train_path, cad_test_path)
    df_cad.to_csv(os.path.join(output_dir, "cad.csv"), index=False)
    print(f"cad.csv               — {df_cad['label'].value_counts().to_dict()}")

    df_gab = load_gab(gab_train_path, gab_test_path)
    df_gab.to_csv(os.path.join(output_dir, "gab.csv"), index=False)
    print(f"gab.csv               — {df_gab['label'].value_counts().to_dict()}")

    print(f"\nAll saved to {output_dir}")


if __name__ == "__main__":
    save_all(
        hatexplain_path="../data/HateXplain.json",
        stg1_path="../data/Implicit_hate/implicit_hate_v1_stg1_posts.tsv",
        stg2_path="../data/Implicit_hate/implicit_hate_v1_stg2_posts.tsv",
        cad_train_path="../data/CAD/cad_v1_train.tsv",
        cad_test_path="../data/CAD/cad_v1_test.tsv",
        gab_train_path="../data/GabHate/ghc_train.tsv",
        gab_test_path="../data/GabHate/ghc_test.tsv",
        output_dir="../data/final_datasets",
    )
