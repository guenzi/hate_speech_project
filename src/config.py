import torch

GROUPS = ["African", "Asian", "Jewish", "Arab", "Caucasian", "Hispanic", "Indian", "Islam"]
MODEL_NAME = "vinai/bertweet-base"
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
