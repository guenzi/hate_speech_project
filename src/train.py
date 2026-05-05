import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.model import BERTweetMultiTask, compute_multitask_loss
from src.dataset import RacismIronyDataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def train_epoch(model, data_loader, optimizer, scheduler, device, lambda_irony):
    """
    Handles the training logic for a single epoch.
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Move tensors to the designated device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        racism_labels = batch["racism_labels"].to(device)
        irony_labels = batch["irony_labels"].to(device)
        
        # Forward pass through the shared BERTweet backbone and task heads
        outputs = model(input_ids, attention_mask)
        
        # Calculate joint loss
        loss, loss_r, loss_i = compute_multitask_loss(
            outputs, 
            racism_labels, 
            irony_labels, 
            lambda_irony=lambda_irony
        )
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate_and_report(model, data_loader, device):
    """
    Detailed evaluation function that provides metrics for the racism detection task.
    This fixes the 'ImportError' in main.py.
    """
    model.eval()
    all_racism_preds = []
    all_racism_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["racism_labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            # Use a 0.5 threshold for binary classification
            preds = (outputs["racism_probs"] > 0.5).int()
            
            all_racism_preds.extend(preds.cpu().numpy().flatten())
            all_racism_labels.extend(labels.cpu().numpy().flatten())
            
    # Calculate and print professional metrics
    print("\n" + "-"*30)
    print("DETAILED PERFORMANCE REPORT")
    print("-"*30)
    print(classification_report(all_racism_labels, all_racism_preds, target_names=["Normal", "Racist"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_racism_labels, all_racism_preds))
    
    # Return F1-score for the summary table in main.py
    return f1_score(all_racism_labels, all_racism_preds)

def main():
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    MAX_LEN = 128
    LAMBDA_IRONY = 0.5 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your preprocessed training data
    # Ensure this path is correct based on your local setup
    train_df = pd.read_csv("data/final_datasets/hatexplain.csv")
    
    train_dataset = RacismIronyDataset(train_df, max_length=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize the Multi-Task model
    model = BERTweetMultiTask().to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, LAMBDA_IRONY)
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Save model weights after each epoch
        save_path = f"bertweet_racism_mtl_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
