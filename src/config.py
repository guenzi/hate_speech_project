import torch

#------------------------------------------------------------------
# GROUPS / RACIAL_TARGETS : ethnic groups tracked for fairness evaluation.
# -  GROUPS is an ordered list used for plots and iteration
# -  RACIAL_TARGETS is a set used for fast membership tests during data loading
# MODEL_NAME : HuggingFace identifier for BERTweet (RoBERTa trained on tweets).
# SEED : global random seed for reproducibility (train/val/test splits, augmentation).
GROUPS = ["African", "Asian", "Jewish", "Arab", "Caucasian", "Hispanic", "Indian", "Islam"]
RACIAL_TARGETS = {"African", "Asian", "Jewish", "Arab", "Caucasian", "Hispanic", "Indian", "Islam"}
MODEL_NAME = "vinai/bertweet-base"
SEED = 42
#------------------------------------------------------------------

def get_device():
    '''
    Get the appropriate device for PyTorch computations (GPU if available, otherwise CPU)

    Returns:
        torch.device: The device to use for PyTorch computations
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
