#!/usr/bin/env python
# coding: utf-8

# # TF-IDF + Log Reg to compare with BERTweet

# In[1]:


import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score


# In[2]:


hatexplain = pd.read_csv("../data/final_datasets/hatexplain.csv")
implicit = pd.read_csv("../data/final_datasets/implicit_hate_test.csv")
cad = pd.read_csv("../data/final_datasets/cad.csv")
gab = pd.read_csv("../data/final_datasets/gab.csv")


# In[3]:


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# In[4]:


df = hatexplain[["text", "label"]].dropna()
df["text"] = df["text"].apply(clean_text)

X_train, X_val, y_train, y_val = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)


# In[5]:


baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1,2)
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

baseline.fit(X_train, y_train)


# In[6]:


preds = baseline.predict(X_val)

print("Accuracy:", accuracy_score(y_val, preds))
print("Macro F1:", f1_score(y_val, preds, average="macro"))
print("Weighted F1:", f1_score(y_val, preds, average="weighted"))

print(classification_report(y_val, preds))


# In[7]:


def evaluate(model, dataset, name):
    data = dataset[["text", "label"]].dropna()
    data["text"] = data["text"].apply(clean_text)

    preds = model.predict(data["text"])

    print(f"\n===== {name} =====")
    print("Macro F1:", f1_score(data["label"], preds, average="macro"))
    print("Weighted F1:", f1_score(data["label"], preds, average="weighted"))


# In[8]:


evaluate(baseline, implicit, "Implicit Hate")
evaluate(baseline, cad, "CAD")
evaluate(baseline, gab, "Gab")

