import pytest
import numpy as np

from qewton.backends.base import DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.algorithms.building_blocks.conv import (
    Conv1D,
    Conv2D,
    Conv3D,
    DoubleConv,
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
    AvgPool1D,
    AvgPool2D,
    AvgPool3D,
    Interpolate,
    BatchNorm1D,
    BatchNorm2D,
    BatchNorm3D,
)


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(DeepLearningBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


# ============================================================================
# Convolution Tests
# ============================================================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_conv1d_initialization(backend):
    """Test Conv1D can be initialized with various parameters."""
    conv = Conv1D(in_channels=3, out_channels=16, kernel_size=3, backend=backend)
    assert conv is not None
    assert conv.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_conv1d_forward(backend, device):
    """Test Conv1D forward pass and output shape."""
    batch_size, in_channels, length = 2, 3, 32
    conv = Conv1D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        padding=1,
        backend=backend,
    )

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, length), device=device
    )

    # Forward pass
    conv.to(device)
    output = conv.forward(x)

    # Check output shape (batch, out_channels, length due to padding)
    assert output.shape == (
        batch_size,
        16,
        length,
    ), f"Expected shape {(batch_size, 16, length)}, got {output.shape}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_conv1d_various_configs(backend, device):
    """Test Conv1D with various configuration options."""
    batch_size, in_channels, length = 2, 3, 64

    # Test with stride
    conv = Conv1D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1,
        backend=backend,
    )
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, length), device=device
    )
    conv.to(device)
    output = conv.forward(x)
    expected_length = (length + 2 * 1 - 3) // 2 + 1
    assert output.shape[2] == expected_length

    # Test without bias
    conv_no_bias = Conv1D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        bias=False,
        padding=1,
        backend=backend,
    )
    conv_no_bias.to(device)
    output = conv_no_bias.forward(x)
    assert output.shape == (batch_size, 16, length)


@pytest.mark.parametrize("backend", BACKENDS)
def test_conv2d_initialization(backend):
    """Test Conv2D can be initialized with various parameters."""
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, backend=backend)
    assert conv is not None
    assert conv.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_conv2d_forward(backend, device):
    """Test Conv2D forward pass and output shape."""
    batch_size, in_channels, height, width = 2, 3, 32, 32
    conv = Conv2D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        padding=1,
        backend=backend,
    )

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, height, width), device=device
    )

    # Forward pass
    conv.to(device)
    output = conv.forward(x)

    # Check output shape
    assert output.shape == (
        batch_size,
        16,
        height,
        width,
    ), f"Expected shape {(batch_size, 16, height, width)}, got {output.shape}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_conv2d_various_configs(backend, device):
    """Test Conv2D with various configuration options."""
    batch_size, in_channels, height, width = 2, 3, 64, 64

    # Test with tuple kernel size
    conv = Conv2D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=(3, 5),
        padding=1,
        backend=backend,
    )
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, height, width), device=device
    )
    conv.to(device)
    output = conv.forward(x)
    assert output.shape == (batch_size, 16, height, width - 2)

    # Test with stride
    conv_stride = Conv2D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1,
        backend=backend,
    )
    conv_stride.to(device)
    output = conv_stride.forward(x)
    expected_h = (height + 2 * 1 - 3) // 2 + 1
    expected_w = (width + 2 * 1 - 3) // 2 + 1
    assert output.shape[2] == expected_h
    assert output.shape[3] == expected_w


@pytest.mark.parametrize("backend", BACKENDS)
def test_conv3d_initialization(backend):
    """Test Conv3D can be initialized with various parameters."""
    conv = Conv3D(in_channels=3, out_channels=16, kernel_size=3, backend=backend)
    assert conv is not None
    assert conv.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_conv3d_forward(backend, device):
    """Test Conv3D forward pass and output shape."""
    batch_size, in_channels, depth, height, width = 2, 3, 16, 32, 32
    conv = Conv3D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        padding=1,
        backend=backend,
    )

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, depth, height, width), device=device
    )

    # Forward pass
    conv.to(device)
    output = conv.forward(x)

    # Check output shape
    assert output.shape == (
        batch_size,
        16,
        depth,
        height,
        width,
    ), f"Expected shape {(batch_size, 16, depth, height, width)}, got {output.shape}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_conv3d_various_configs(backend, device):
    """Test Conv3D with various configuration options."""
    batch_size, in_channels, depth, height, width = 2, 3, 16, 32, 32

    # Test with tuple kernel size
    conv = Conv3D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=(3, 3, 5),
        padding=1,
        backend=backend,
    )
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, depth, height, width), device=device
    )
    conv.to(device)
    output = conv.forward(x)
    assert output.shape == (batch_size, 16, depth, height, width - 2)

    # Test with stride
    conv_stride = Conv3D(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1,
        backend=backend,
    )
    conv_stride.to(device)
    output = conv_stride.forward(x)
    expected_d = (depth + 2 * 1 - 3) // 2 + 1
    expected_h = (height + 2 * 1 - 3) // 2 + 1
    expected_w = (width + 2 * 1 - 3) // 2 + 1
    assert output.shape[2] == expected_d
    assert output.shape[3] == expected_h
    assert output.shape[4] == expected_w


@pytest.mark.parametrize("backend", BACKENDS)
def test_doubleconv_initialization(backend):
    """Test DoubleConv can be initialized with various parameters."""
    conv = DoubleConv(in_channels=3, out_channels=16, kernel_size=3, backend=backend)
    assert conv is not None
    assert conv.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_doubleconv_forward_1d(backend, device):
    """Test DoubleConv forward pass for 1D data."""
    batch_size, in_channels, length = 2, 3, 32
    conv = DoubleConv(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        padding=1,
        backend=backend,
    )

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, length), device=device
    )

    # Forward pass
    conv.to(device)
    output = conv.forward(x)

    # Check output shape (should have out_channels, not in_channels)
    assert output.shape == (batch_size, 16, length)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_doubleconv_forward_2d(backend, device):
    """Test DoubleConv forward pass for 2D data."""
    batch_size, in_channels, height, width = 2, 3, 32, 32
    conv = DoubleConv(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=(3, 3),
        padding=1,
        backend=backend,
    )

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, in_channels, height, width), device=device
    )

    # Forward pass
    conv.to(device)
    output = conv.forward(x)

    # Check output shape
    assert output.shape == (batch_size, 16, height, width)


# ============================================================================
# MaxPooling Tests
# ============================================================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_maxpool1d_initialization(backend):
    """Test MaxPool1D can be initialized with various parameters."""
    pool = MaxPool1D(kernel_size=2, backend=backend)
    assert pool is not None
    assert pool.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_maxpool1d_forward(backend, device):
    """Test MaxPool1D forward pass and output shape."""
    batch_size, channels, length = 2, 3, 32
    pool = MaxPool1D(kernel_size=2, stride=2, backend=backend)

    # Create input tensor
    x = backend.build_tensor(np.random.randn(batch_size, channels, length), device=device)

    # Forward pass
    output = pool.forward(x)

    # Check output shape (length should be halved with stride=2)
    expected_length = (length - 2) // 2 + 1
    assert output.shape == (batch_size, channels, expected_length)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_maxpool1d_various_configs(backend, device):
    """Test MaxPool1D with various configurations."""
    batch_size, channels, length = 2, 3, 64

    # Test with padding
    pool = MaxPool1D(kernel_size=3, stride=1, padding=1, backend=backend)
    x = backend.build_tensor(np.random.randn(batch_size, channels, length), device=device)
    output = pool.forward(x)
    assert output.shape == (batch_size, channels, length)


@pytest.mark.parametrize("backend", BACKENDS)
def test_maxpool2d_initialization(backend):
    """Test MaxPool2D can be initialized with various parameters."""
    pool = MaxPool2D(kernel_size=2, backend=backend)
    assert pool is not None
    assert pool.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_maxpool2d_forward(backend, device):
    """Test MaxPool2D forward pass and output shape."""
    batch_size, channels, height, width = 2, 3, 32, 32
    pool = MaxPool2D(kernel_size=2, stride=2, backend=backend)

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, height, width), device=device
    )

    # Forward pass
    output = pool.forward(x)

    # Check output shape
    expected_h = (height - 2) // 2 + 1
    expected_w = (width - 2) // 2 + 1
    assert output.shape == (batch_size, channels, expected_h, expected_w)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_maxpool2d_various_configs(backend, device):
    """Test MaxPool2D with various configurations."""
    batch_size, channels, height, width = 2, 3, 64, 64

    # Test with tuple kernel size
    pool = MaxPool2D(kernel_size=(2, 3), stride=(2, 3), backend=backend)
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, height, width), device=device
    )
    output = pool.forward(x)
    expected_h = (height - 2) // 2 + 1
    expected_w = (width - 3) // 3 + 1
    assert output.shape == (batch_size, channels, expected_h, expected_w)


@pytest.mark.parametrize("backend", BACKENDS)
def test_maxpool3d_initialization(backend):
    """Test MaxPool3D can be initialized with various parameters."""
    pool = MaxPool3D(kernel_size=2, backend=backend)
    assert pool is not None
    assert pool.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_maxpool3d_forward(backend, device):
    """Test MaxPool3D forward pass and output shape."""
    batch_size, channels, depth, height, width = 2, 3, 16, 32, 32
    pool = MaxPool3D(kernel_size=2, stride=2, backend=backend)

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, depth, height, width), device=device
    )

    # Forward pass
    output = pool.forward(x)

    # Check output shape
    expected_d = (depth - 2) // 2 + 1
    expected_h = (height - 2) // 2 + 1
    expected_w = (width - 2) // 2 + 1
    assert output.shape == (batch_size, channels, expected_d, expected_h, expected_w)


# ============================================================================
# AvgPooling Tests
# ============================================================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_avgpool1d_initialization(backend):
    """Test AvgPool1D can be initialized with various parameters."""
    pool = AvgPool1D(kernel_size=2, backend=backend)
    assert pool is not None
    assert pool.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_avgpool1d_forward(backend, device):
    """Test AvgPool1D forward pass and output shape."""
    batch_size, channels, length = 2, 3, 32
    pool = AvgPool1D(kernel_size=2, stride=2, backend=backend)

    # Create input tensor
    x = backend.build_tensor(np.random.randn(batch_size, channels, length), device=device)

    # Forward pass
    output = pool.forward(x)

    # Check output shape
    expected_length = (length - 2) // 2 + 1
    assert output.shape == (batch_size, channels, expected_length)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_avgpool1d_various_configs(backend, device):
    """Test AvgPool1D with various configurations."""
    batch_size, channels, length = 2, 3, 64

    # Test with padding
    pool = AvgPool1D(kernel_size=3, stride=1, padding=1, backend=backend)
    x = backend.build_tensor(np.random.randn(batch_size, channels, length), device=device)
    output = pool.forward(x)
    assert output.shape == (batch_size, channels, length)


@pytest.mark.parametrize("backend", BACKENDS)
def test_avgpool2d_initialization(backend):
    """Test AvgPool2D can be initialized with various parameters."""
    pool = AvgPool2D(kernel_size=2, backend=backend)
    assert pool is not None
    assert pool.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_avgpool2d_forward(backend, device):
    """Test AvgPool2D forward pass and output shape."""
    batch_size, channels, height, width = 2, 3, 32, 32
    pool = AvgPool2D(kernel_size=2, stride=2, backend=backend)

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, height, width), device=device
    )

    # Forward pass
    output = pool.forward(x)

    # Check output shape
    expected_h = (height - 2) // 2 + 1
    expected_w = (width - 2) // 2 + 1
    assert output.shape == (batch_size, channels, expected_h, expected_w)


@pytest.mark.parametrize("backend", BACKENDS)
def test_avgpool3d_initialization(backend):
    """Test AvgPool3D can be initialized with various parameters."""
    pool = AvgPool3D(kernel_size=2, backend=backend)
    assert pool is not None
    assert pool.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_avgpool3d_forward(backend, device):
    """Test AvgPool3D forward pass and output shape."""
    batch_size, channels, depth, height, width = 2, 3, 16, 32, 32
    pool = AvgPool3D(kernel_size=2, stride=2, backend=backend)

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, depth, height, width), device=device
    )

    # Forward pass
    output = pool.forward(x)

    # Check output shape
    expected_d = (depth - 2) // 2 + 1
    expected_h = (height - 2) // 2 + 1
    expected_w = (width - 2) // 2 + 1
    assert output.shape == (batch_size, channels, expected_d, expected_h, expected_w)


# ============================================================================
# BatchNormalization Tests
# ============================================================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_batchnorm1d_initialization(backend):
    """Test BatchNorm1D can be initialized with various parameters."""
    bn = BatchNorm1D(num_features=16, backend=backend)
    assert bn is not None
    assert bn.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_batchnorm1d_forward(backend, device):
    """Test BatchNorm1D forward pass and output shape."""
    batch_size, num_features, length = 4, 16, 32
    bn = BatchNorm1D(num_features=num_features, backend=backend)

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, num_features, length), device=device
    )

    # Forward pass
    bn.to(device)
    output = bn.forward(x)

    # Check output shape
    assert output.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_batchnorm1d_various_configs(backend, device):
    """Test BatchNorm1D with various configurations."""
    batch_size, num_features, length = 4, 16, 32

    # Test with weight and bias
    bn = BatchNorm1D(num_features=num_features, weight=True, bias=True, backend=backend)
    x = backend.build_tensor(
        np.random.randn(batch_size, num_features, length), device=device
    )
    bn.to(device)
    output = bn.forward(x)
    assert output.shape == x.shape

    # Test with different eps and momentum
    bn = BatchNorm1D(num_features=num_features, eps=1e-3, momentum=0.05, backend=backend)
    bn.to(device)
    output = bn.forward(x)
    assert output.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_batchnorm2d_initialization(backend):
    """Test BatchNorm2D can be initialized with various parameters."""
    bn = BatchNorm2D(num_features=16, backend=backend)
    assert bn is not None
    assert bn.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_batchnorm2d_forward(backend, device):
    """Test BatchNorm2D forward pass and output shape."""
    batch_size, num_features, height, width = 4, 16, 32, 32
    bn = BatchNorm2D(num_features=num_features, backend=backend)

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, num_features, height, width), device=device
    )

    # Forward pass
    bn.to(device)
    output = bn.forward(x)

    # Check output shape
    assert output.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_batchnorm2d_various_configs(backend, device):
    """Test BatchNorm2D with various configurations."""
    batch_size, num_features, height, width = 4, 16, 32, 32

    # Test with weight and bias
    bn = BatchNorm2D(num_features=num_features, weight=True, bias=True, backend=backend)
    x = backend.build_tensor(
        np.random.randn(batch_size, num_features, height, width), device=device
    )
    bn.to(device)
    output = bn.forward(x)
    assert output.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_batchnorm3d_initialization(backend):
    """Test BatchNorm3D can be initialized with various parameters."""
    bn = BatchNorm3D(num_features=16, backend=backend)
    assert bn is not None
    assert bn.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_batchnorm3d_forward(backend, device):
    """Test BatchNorm3D forward pass and output shape."""
    batch_size, num_features, depth, height, width = 4, 16, 16, 32, 32
    bn = BatchNorm3D(num_features=num_features, backend=backend)

    # Create input tensor
    x = backend.build_tensor(
        np.random.randn(batch_size, num_features, depth, height, width), device=device
    )

    # Forward pass
    bn.to(device)
    output = bn.forward(x)

    # Check output shape
    assert output.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_batchnorm3d_various_configs(backend, device):
    """Test BatchNorm3D with various configurations."""
    batch_size, num_features, depth, height, width = 4, 16, 16, 32, 32

    # Test with weight and bias
    bn = BatchNorm3D(num_features=num_features, weight=True, bias=True, backend=backend)
    x = backend.build_tensor(
        np.random.randn(batch_size, num_features, depth, height, width), device=device
    )
    bn.to(device)
    output = bn.forward(x)
    assert output.shape == x.shape


# ============================================================================
# Interpolation Tests
# ============================================================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_interpolate_initialization(backend):
    """Test Interpolate can be initialized with size or scale factor."""
    interp = Interpolate(size=10, backend=backend)
    assert interp is not None
    assert interp.backend == backend

    interp_scale = Interpolate(scale_factor=2, backend=backend)
    assert interp_scale is not None
    assert interp_scale.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_interpolate_forward_size_1d(backend, device):
    """Test Interpolate forward pass using a 1D output size."""
    batch_size, channels, length = 2, 3, 8
    interp = Interpolate(size=16, backend=backend)
    x = backend.build_tensor(np.random.randn(batch_size, channels, length), device=device)
    output = interp.forward(x)
    assert output.shape == (batch_size, channels, 16)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_interpolate_forward_scale_factor_2d(backend, device):
    """Test Interpolate forward pass using a 2D scale factor."""
    batch_size, channels, height, width = 2, 3, 8, 8
    interp = Interpolate(scale_factor=(2, 2), backend=backend)
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, height, width), device=device
    )
    output = interp.forward(x)
    assert output.shape == (batch_size, channels, height * 2, width * 2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_interpolate_forward_bilinear_align_corners(backend, device):
    """Test Interpolate forward pass with bilinear mode and align_corners."""
    batch_size, channels, height, width = 2, 3, 8, 8
    interp = Interpolate(
        size=(16, 16), interpolate_mode="bilinear", align_corners=True, backend=backend
    )
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, height, width), device=device
    )
    output = interp.forward(x)
    assert output.shape == (batch_size, channels, 16, 16)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_interpolate_forward_size_3d(backend, device):
    """Test Interpolate forward pass using a 3D output size."""
    batch_size, channels, depth, height, width = 2, 3, 4, 8, 8
    interp = Interpolate(size=(depth, 16, 16), backend=backend)
    x = backend.build_tensor(
        np.random.randn(batch_size, channels, depth, height, width), device=device
    )
    output = interp.forward(x)
    assert output.shape == (batch_size, channels, depth, 16, 16)
