import os
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_torch_file(path: str, map_location="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location=map_location)