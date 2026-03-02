from pathlib import Path
import torch

def get_project_root(markers=("Classes", "Data")):
    here = Path(__file__).resolve().parent

    for p in [here] + list(here.parents):
        if any((p / m).exists() for m in markers):
            return p

    raise RuntimeError("Project root not found")


def get_device():
    # Set device for PyTorch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    return device
