import torch
from qewton.config.devices import CPU, CUDA, Device


def get_torch_device(device: Device):
    if isinstance(device, CPU):
        return torch.device("cpu")
    elif isinstance(device, CUDA):
        return torch.device(f"cuda:{device.index}")
