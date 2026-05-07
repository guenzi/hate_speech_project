import ast
import json
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import SEED, RACIAL_TARGETS

def clean_tweet(text: str) -> str:
    text = str(text)
    text = re.sub(r"<user>", "@USER", text)
    text = re.sub(r"<url>", "HTTPURL", text)
    text = re.sub(r"http\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\[linebreak\]", " ", text)
    text = text.strip()
    return text
    
def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "targets" in df.columns:
            df["targets"] = df["targets"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return df

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
            "label": is_racist,
        })

    df = pd.DataFrame(rows)
    df = df[(df["label"] == 1) | (df["majority_label"] == "normal")].copy()
    df = df[df["targets"].apply(lambda t: len(set(t) & RACIAL_TARGETS) == 1)].reset_index(drop=True)

    return df[["id", "text", "label", "targets"]]

def split_data(df):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df_train, df_temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["label"])
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, sample_weights=None):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.sample_weights is not None:
            item["weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return item

def make_loaders(df_tr, df_v, df_te, tokenizer, max_len, batch_size, sample_weights=None):
    return (
        DataLoader(
            TweetDataset(df_tr["text"], df_tr["label"], tokenizer, max_len, sample_weights),
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            TweetDataset(df_v["text"], df_v["label"], tokenizer, max_len),
            batch_size=batch_size,
        ),
        DataLoader(
            TweetDataset(df_te["text"], df_te["label"], tokenizer, max_len),
            batch_size=batch_size,
        ),
    )