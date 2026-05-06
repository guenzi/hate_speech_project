import ast

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import SEED


def load_data(data_path):
    df = pd.read_csv(data_path)
    df["targets_parsed"] = df["targets"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return df


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
