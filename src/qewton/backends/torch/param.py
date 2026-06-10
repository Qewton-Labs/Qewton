import torch
from qewton.backends.param import ParameterBackend


class TorchParameterBackend(ParameterBackend[torch.Tensor]):
    @staticmethod
    def initialize(shape=None, tensor=None) -> torch.Tensor:
        if tensor is not None:
            assert isinstance(
                tensor, torch.Tensor
            ), "Torch can only work with torch.Tensors, but got {type(tensor)} instead."
            param = torch.nn.Parameter(tensor)
        elif shape is not None:
            # TODO: We need some kind of initialization for these parameters
            # E.g. 0, rand, xavier,... But this also needs to be exposed to the outside
            param = torch.nn.Parameter(torch.zeros(shape), requires_grad=True)
            if len(shape) > 1:
                torch.nn.init.xavier_uniform_(param)
        else:
            raise ValueError("Either 'shape' or 'tensor' must be provided to initialize.")
        return param

    @staticmethod
    def to(data, device) -> torch.Tensor:
        return data.data.to(device)

    @staticmethod
    def requires_grad(data, requires_grad: bool) -> torch.Tensor:
        data.requires_grad = requires_grad
        return data
