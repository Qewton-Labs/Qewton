import pytest

from qewton.backends.base import DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.algorithms.dl_models.cnn import UNet


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(DeepLearningBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_unet_even_kernel_size(backend):
    with pytest.raises(AssertionError):
        _ = UNet(1, (8, 16, 32), 2, conv_kernel_size=(2,), backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unet_initialization_1d(backend):
    unet = UNet(1, (8, 16, 32), 2, conv_kernel_size=(3,), backend=backend)
    assert len(unet.channels) == 3
    assert unet.in_channels.current_value == 1
    assert unet.out_channels.current_value == 2
    assert unet.in_channels.is_fixed
    assert unet.out_channels.is_fixed


@pytest.mark.parametrize("backend", BACKENDS)
def test_unet_initialization_2d(backend):
    unet = UNet(1, (16, 32), 3, conv_kernel_size=(3, 3), backend=backend)
    assert len(unet.channels) == 2
    assert unet.in_channels.current_value == 1
    assert unet.out_channels.current_value == 3
    assert unet.conv_kernel_size.current_value == (3, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unet_initialization_3d(backend):
    unet = UNet(2, (8, 16, 32), 4, conv_kernel_size=(3, 3, 3), backend=backend)
    assert len(unet.channels) == 3
    assert unet.in_channels.current_value == 2
    assert unet.out_channels.current_value == 4
    assert unet.conv_kernel_size.current_value == (3, 3, 3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_unet_1d_eval(backend, device):
    unet = UNet(1, (8, 16, 32), 2, conv_kernel_size=(3,), backend=backend)
    points = backend.math.zeros((10, 1, 64), device=device)
    unet.to(device=device)
    output = unet(points)
    assert output.shape == (10, 2, 64)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_unet_2d_eval(backend, device):
    unet = UNet(1, (16, 32), 3, conv_kernel_size=(3, 3), backend=backend)
    points = backend.math.zeros((4, 1, 64, 64), device=device)
    unet.to(device=device)
    output = unet(points)
    assert output.shape == (4, 3, 64, 64)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_unet_3d_eval(backend, device):
    unet = UNet(2, (8, 16, 32), 4, conv_kernel_size=(3, 3, 3), backend=backend)
    points = backend.math.zeros((2, 2, 32, 32, 32), device=device)
    unet.to(device=device)
    output = unet(points)
    assert output.shape == (2, 4, 32, 32, 32)
