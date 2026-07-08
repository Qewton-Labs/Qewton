import torch
from qewton.backends.param import ParameterBackend
from qewton.backends.torch.device import get_torch_device


class TorchParameterBackend(ParameterBackend[torch.Tensor]):
    @staticmethod
    def initialize(shape=None, tensor=None) -> torch.Tensor:
        if tensor is not None:
            assert isinstance(
                tensor, torch.Tensor
            ), "Torch can only work with torch.Tensors, but got {type(tensor)} instead."
            param = tensor.requires_grad_(True)
        elif shape is not None:
            # TODO: We need some kind of initialization for these parameters
            # E.g. 0, rand, xavier,... But this also needs to be exposed to the outside
            param = torch.zeros(shape, requires_grad=True)
            if len(shape) > 1:
                torch.nn.init.xavier_uniform_(param)
        else:
            raise ValueError("Either 'shape' or 'tensor' must be provided to initialize.")
        return param

    @staticmethod
    def to(data: torch.Tensor, device) -> torch.Tensor:
        new_data = data.to(get_torch_device(device=device)).detach()
        new_data.requires_grad = data.requires_grad
        return new_data

    @staticmethod
    def requires_grad(data, requires_grad: bool) -> torch.Tensor:
        data.requires_grad = requires_grad
        return data
