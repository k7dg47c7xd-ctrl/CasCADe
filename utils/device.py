import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(dtype_name: str):
    dtype_name = dtype_name.lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")