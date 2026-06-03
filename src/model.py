import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup

from config import MODEL_NAME

class FocalLoss(nn.Module):
    '''
    Focal Loss for addressing class imbalance in classification tasks
    '''
    def __init__(self, gamma=2.0, reduction="none"):
        '''
        Initialize the Focal Loss module

        Args:
            gamma (float): Focusing parameter that reduces the relative loss for well-classified examples
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        
        Returns:
            None
        '''
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        '''
        Compute the Focal Loss between logits and labels

        Args:
            logits (torch.Tensor): The predicted logits from the model (shape: [batch_size, num_classes])
            labels (torch.Tensor): The true labels (shape: [batch_size])

        Returns:
            torch.Tensor: The computed Focal Loss
        '''
        ce = nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        return loss


def train_bertweet(train_loader, val_loader, device, epochs, lr,
                   use_sample_weights=False, patience=2):
    '''
    Train the RoBERTa model for sequence classification using the provided training and validation data loaders

    Args:
        train_loader (DataLoader): DataLoader for the training data
        val_loader (DataLoader): DataLoader for the validation data
        device (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu')
        epochs (int): The number of training epochs
        lr (float): The learning rate for the optimizer
        use_sample_weights (bool): Whether to use sample weights during training
        patience (int): Number of epochs to wait for improvement before early stopping
    
    Returns:
        model (RobertaForSequenceClassification): The trained RoBERTa model
        history (dict): A dictionary containing training loss and validation F1 scores for each epoch
    '''
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    focal_loss = FocalLoss(gamma=2.0, reduction="none")
    history = {"train_loss": [], "val_f1": []}

    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = focal_loss(logits, batch["label"].to(device))
            if use_sample_weights and "weight" in batch:
                loss = (loss * batch["weight"].to(device)).mean()
            else:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                preds.extend(logits.argmax(-1).cpu().numpy())
                trues.extend(batch["label"].numpy())

        val_f1 = f1_score(trues, preds, average="macro")
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"   best model (val F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"   no improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"   early stopping at epoch {epoch+1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, history


def predict(model, loader, device):
    '''
    Generate predictions from the model for the given data loader

    Args:
        model: The trained model
        loader (DataLoader): DataLoader for the data to predict on
        device (torch.device): The device to run the prediction on (e.g., 'cuda')
    
    Returns:
        np.array: An array of predicted labels for the input data
    '''
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds.extend(logits.argmax(-1).cpu().numpy())
    return np.array(preds)


def save_checkpoint(model, path):
    '''
    Save the model checkpoint to the specified path including the model's state_dict

    Args:
        model: The trained model to save
        path (str): The file path to save the model checkpoint

    Returns:
        None
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(filepath, device):
    """
    Load a model checkpoint from the specified file path and map the weights to the RoBERTa architecture for sequence classification

    Args:
        filepath (str): The file path of the checkpoint to load
        device (torch.device): The device to load the model onto (e.g., 'cuda' or 'cpu')

    Returns:
        model: The loaded model with weights mapped to the classification architecture
    """
    print(f"Loading checkpoint: {filepath}")
    
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=True)
    print("-> Weights successfully mapped to the classification architecture!")
        
    model.to(device)
    model.eval()
    return model