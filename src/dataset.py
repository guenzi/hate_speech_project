import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class RacismIronyDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name="vinai/bertweet-base", max_length=128):
        """
        Args:
            dataframe: Pandas DataFrame contenant les colonnes 'text' et 'label' (racisme).
            tokenizer_name: Nom du modèle pour charger le tokenizer BERTweet.
            max_length: Longueur maximale des séquences de tokens.
        """
        self.data = dataframe
        # BERTweet nécessite l'argument normalization=True pour traiter @USER et HTTPURL
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, normalization=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row["text"])
        
        # Label principal : Racisme
        racism_label = int(row["label"])
        
        # Label auxiliaire : Ironie
        # Si la colonne 'irony_label' n'existe pas dans le CSV (ex: HateXplain), on met -1
        irony_label = int(row.get("irony_label", -1))

        # Tokenisation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "racism_labels": torch.tensor(racism_label, dtype=torch.long),
            "irony_labels": torch.tensor(irony_label, dtype=torch.long)
        }
