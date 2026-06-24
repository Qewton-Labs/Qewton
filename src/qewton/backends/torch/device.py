import torch
from qewton.config.devices import CPU, CUDA, Device


def get_torch_device(device: Device | str):
    if isinstance(device, CPU):
        return torch.device("cpu")
    if isinstance(device, CUDA):
        return torch.device(f"cuda:{device.index}")
    if isinstance(device, str):
        return device
