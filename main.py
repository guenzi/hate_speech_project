import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Internal imports
from src.preprocess import save_all
from src.train import main as train_pipeline, evaluate_and_report
from src.model import BERTweetMultiTask
from src.dataset import RacismIronyDataset

def run_pipeline(args):
    # 1. Data Preprocessing Phase
    # This prepares the raw datasets into cleaned CSV files
    if args.preprocess:
        print("--- Starting Preprocessing ---")
        save_all(
            hatexplain_path=args.hx_path,
            stg1_path=args.stg1_path,
            stg2_path=args.stg2_path,
            cad_train_path=args.cad_train,
            cad_test_path=args.cad_test,
            gab_train_path=args.gab_train,
            gab_test_path=args.gab_test,
            output_dir=args.output_dir
        )
        print("--- Preprocessing Complete ---")

    # 2. Training Phase
    # Orchestrates the Multi-Task learning process
    if args.train:
        print("--- Starting Multi-Task Training ---")
        train_pipeline() 
        print("--- Training Complete ---")

    # 3. Detailed Evaluation Phase (New Metrics Feature)
    # Analyzes performance across multiple datasets using F1-score and Confusion Matrix
    if args.eval:
        print("--- Starting Detailed Evaluation ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model architecture
        model = BERTweetMultiTask()
        
        # Load saved weights if they exist
        if os.path.exists(args.model_path):
            print(f"Loading weights from {args.model_path}...")
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        else:
            print(f"Warning: Model file {args.model_path} not found. Proceeding with random weights.")
            
        model.to(device).eval()

        # List of preprocessed test files to analyze
        test_files = ["hatexplain.csv", "cad.csv", "gab.csv", "implicit_hate_test.csv"]
        
        evaluation_results = {}
        
        for file_name in test_files:
            file_path = os.path.join(args.output_dir, file_name)
            if os.path.exists(file_path):
                print(f"\n" + "="*40)
                print(f" ANALYZING: {file_name} ".center(40, "="))
                print("="*40)
                
                # Load data into DataFrame
                df = pd.read_csv(file_path)
                
                # Initialize Dataset and DataLoader
                test_ds = RacismIronyDataset(df)
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
                
                # Execute measurement function (from src/train.py)
                # It computes F1, Precision, Recall, and Confusion Matrix
                f1 = evaluate_and_report(model, test_loader, device)
                evaluation_results[file_name] = f1
            else:
                print(f"\n[Skipping] {file_name}: File not found in {args.output_dir}")

        # Final Summary Printout
        print("\n" + " GLOBAL EVALUATION SUMMARY ".center(40, "-"))
        for name, score in evaluation_results.items():
            print(f"{name:25} | F1-Score: {score:.4f}")
        print("-" * 40)

    # 4. Single Sample Inference
    # Quick test to verify model behavior on a specific sentence
    if args.test_sample:
        print("\n--- Running Test Inference ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = BERTweetMultiTask()
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device).eval()
        
        # Example of potentially ironic/hateful content
        sample_text = "Great, another immigrant stealing our jobs... said nobody with a brain."
        
        # Tokenization via Dataset wrapper
        ds = RacismIronyDataset(pd.DataFrame([{"text": sample_text, "label": 0}]))
        item = ds[0]
        
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        mask = item["attention_mask"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, mask)
            racism_prob = outputs["racism_probs"].item()
            irony_prob = outputs["irony_probs"].item()
            
        print(f"Input Text: {sample_text}")
        print(f"Racism Probability: {racism_prob:.4f}")
        print(f"Irony Probability: {irony_prob:.4f}")
        print(f"Classification: {'[RACIST]' if racism_prob > 0.5 else '[NORMAL]'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twitter Racism Detection Multi-Task Pipeline")
    
    # Preprocessing arguments
    parser.add_argument("--preprocess", action="store_true", help="Run data cleaning and preparation")
    parser.add_argument("--hx_path", default="data/HateXplain.json")
    parser.add_argument("--stg1_path", default="data/Implicit_hate/implicit_hate_v1_stg1_posts.tsv")
    parser.add_argument("--stg2_path", default="data/Implicit_hate/implicit_hate_v1_stg2_posts.tsv")
    parser.add_argument("--cad_train", default="data/CAD/cad_v1_train.tsv")
    parser.add_argument("--cad_test", default="data/CAD/cad_v1_test.tsv")
    parser.add_argument("--gab_train", default="data/GabHate/ghc_train.tsv")
    parser.add_argument("--gab_test", default="data/GabHate/ghc_test.tsv")
    parser.add_argument("--output_dir", default="data/final_datasets")
    
    # Training and Evaluation arguments
    parser.add_argument("--train", action="store_true", help="Run the model training")
    parser.add_argument("--eval", action="store_true", help="Run deep analysis on test sets")
    parser.add_argument("--model_path", default="bertweet_racism_mtl_epoch3.pt", help="Path to trained model weights")
    
    # Inference arguments
    parser.add_argument("--test_sample", action="store_true", help="Predict on a single hard-coded example")

    args = parser.parse_args()
    run_pipeline(args)
