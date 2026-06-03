import torch

#------------------------------------------------------------------
# Configuration of everything for the hate speech detection of this project
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
