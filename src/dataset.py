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
    '''
    Clean the tweet text by replacing user mentions, URLs, and line breaks with standardized tokens and removing extra whitespace

    Args:
        text (str): The original tweet text
    
    Returns:
        str: The cleaned tweet text
    '''
    text = str(text)
    text = re.sub(r"<user>", "@USER", text)
    text = re.sub(r"<url>", "HTTPURL", text)
    text = re.sub(r"http\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\[linebreak\]", " ", text)
    text = text.strip()
    return text
    
def load_data(path: str) -> pd.DataFrame:
    '''
    Load the dataset from a CSV or JSON file, clean the text, and prepare it for training by filtering and structuring the data appropriately

    Args:
        path (str): The file path to the dataset (CSV or JSON)
    
    Returns:
        pd.DataFrame: A DataFrame containing the cleaned and structured dataset with columns for id, text, label, and targets
    '''
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
    '''
    Split the dataset into training, validation, and test sets

    Args:
        df (pd.DataFrame): The input dataset

    Returns:
        tuple: A tuple containing the training, validation, and test datasets
    '''
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
    '''
    A custom PyTorch Dataset class for handling tweet data, including tokenization and preparation of input features for model training
    '''
    def __init__(self, texts, labels, tokenizer, max_len, sample_weights=None):
        '''
        Initialize the TweetDataset with texts, labels, tokenizer, maximum sequence length, and optional sample weights

        Args:
            texts (list): A list of tweet texts
            labels (list): A list of corresponding labels for the tweets
            tokenizer: A tokenizer object for encoding the tweet texts
            max_len (int): The maximum sequence length for tokenization
            sample_weights (list, optional): A list of sample weights for handling class imbalance (default: None)

        Returns:
            None
        '''
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sample_weights = sample_weights

    def __len__(self):
        '''
        Return the number of samples in the dataset

        Returns:
            int: The number of samples in the dataset
        '''
        return len(self.texts)

    def __getitem__(self, idx):
        '''
        Retrieve a single sample from the dataset at the specified index, including tokenized input features and label

        Args:
            idx (int): The index of the sample to retrieve

        Returns:
            dict: A dictionary containing the tokenized input features and label for the specified sample
        '''
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
    '''
    Create DataLoader objects for the training, validation, and test datasets

    Args:
        df_tr (pd.DataFrame): The training dataset
        df_v (pd.DataFrame): The validation dataset
        df_te (pd.DataFrame): The test dataset
        tokenizer: A tokenizer object for encoding the tweet texts
        max_len (int): The maximum sequence length for tokenization
        batch_size (int): The batch size for the DataLoader
        sample_weights (list, optional): A list of sample weights for handling class imbalance in the training dataset (default: None)
    
    Returns:
        tuple: A tuple containing the DataLoader objects for the training, validation, and test datasets
    '''
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