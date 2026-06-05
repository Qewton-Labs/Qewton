from typing import Literal

import torch
from qewton.backends.nn import NNBackend


class TorchNNBackend(NNBackend[torch.Tensor]):
    """Torch implementations of neural network operations."""

    relu = torch.nn.functional.relu
    sigmoid = torch.nn.functional.sigmoid
    tanh = torch.nn.functional.tanh
    softmax = torch.nn.functional.softmax
    gelu = torch.nn.functional.gelu

    conv1d = torch.nn.functional.conv1d
    conv2d = torch.nn.functional.conv2d
    conv3d = torch.nn.functional.conv3d

    max_pool1d = torch.nn.functional.max_pool1d
    max_pool2d = torch.nn.functional.max_pool2d
    max_pool3d = torch.nn.functional.max_pool3d

    avg_pool1d = torch.nn.functional.avg_pool1d
    avg_pool2d = torch.nn.functional.avg_pool2d
    avg_pool3d = torch.nn.functional.avg_pool3d

    batch_norm1d = torch.nn.functional.batch_norm
    batch_norm2d = torch.nn.functional.batch_norm
    batch_norm3d = torch.nn.functional.batch_norm

    @staticmethod
    def interpolate(
        x: torch.Tensor,
        size: int | tuple[int] | tuple[int, int] | tuple[int, int, int] | None = None,
        scale_factor: float | tuple[float] | None = None,
        mode: Literal[
            "nearest", "linear", "bilinear", "bicubic", "trilinear"
        ] = "nearest",
        align_corners: bool = False,
    ) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, size, scale_factor, mode, align_corners)
