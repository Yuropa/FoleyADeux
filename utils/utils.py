import torch

def create_device():
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = "auto"
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
    else:
        device = "cpu"
        torch_dtype = "auto"

    return (device, torch_dtype)
