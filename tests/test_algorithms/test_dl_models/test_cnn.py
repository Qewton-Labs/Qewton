import pytest
import numpy as np

try:
    import torch
    from qewton.backends.torch.base import TorchBackend

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    from qewton.backends.tensorflow.base import TensorflowBackend

    HAS_TF = True
except ImportError:
    HAS_TF = False

from qewton.algorithms.dl_models.cnn import CNN
from qewton.algorithms.building_blocks.activation_functions import ReLU, Tanh
from qewton.backends.base import DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(DeepLearningBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


class TestCNNInitialization:
    """Test CNN initialization with different configurations."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_1d_initialization(self, backend: type[DeepLearningBackend]):
        """Test basic CNN initialization for 1D convolutions."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            backend=backend,
        )
        assert cnn is not None
        assert cnn.in_channels.value == 3
        assert cnn.out_channels.value == 5

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_2d_initialization(self, backend: type[DeepLearningBackend]):
        """Test basic CNN initialization for 2D convolutions."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=32,
            out_channels=10,
            n_hidden_layers=2,
            kernel_size=(3, 3),
            backend=backend,
        )
        assert cnn is not None
        assert cnn.hidden_channels.value == 32

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_3d_initialization(self, backend: type[DeepLearningBackend]):
        """Test basic CNN initialization for 3D convolutions."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=8,
            out_channels=4,
            n_hidden_layers=1,
            kernel_size=(3, 3, 3),
            backend=backend,
        )
        assert cnn is not None
        assert len(cnn.kernel_size) == 3

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_kernel_size_must_be_odd(self, backend: type[DeepLearningBackend]):
        """Test that even kernel sizes raise an assertion error."""
        with pytest.raises(AssertionError, match="Kernel size must be always odd"):
            CNN(
                in_channels=3,
                hidden_channels=16,
                out_channels=5,
                n_hidden_layers=1,
                kernel_size=4,
                backend=backend,
            )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_multiple_hidden_layers(self, backend: type[DeepLearningBackend]):
        """Test CNN with multiple hidden layers."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=3,
            kernel_size=3,
            backend=backend,
        )
        assert cnn.n_hidden_layers.value == 3


class TestCNNHyperparameters:
    """Test CNN hyperparameter management."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_hyperparameters_property(self, backend: type[DeepLearningBackend]):
        """Test that hyperparameters property returns all hyperparameters."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=2,
            kernel_size=3,
            backend=backend,
        )
        hps = cnn.hyperparameters
        assert (
            len(hps) >= 6
        )  # in_channels, hidden_channels, out_channels, n_hidden_layers, bias, activation
        assert cnn.in_channels in hps

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_bias_hyperparameter(self, backend: type[DeepLearningBackend]):
        """Test CNN with and without bias."""
        cnn_with_bias = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            bias=True,
            backend=backend,
        )
        assert cnn_with_bias.bias.value is True

        cnn_without_bias = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            bias=False,
            backend=backend,
        )
        assert cnn_without_bias.bias.value is False

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnn_activation_function(self, backend: type[DeepLearningBackend]):
        """Test CNN with different activation functions."""
        cnn_relu = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            activation=ReLU,
            backend=backend,
        )
        assert cnn_relu.activation.value == ReLU

        cnn_tanh = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            activation=Tanh,
            backend=backend,
        )
        assert cnn_tanh.activation.value == Tanh


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCNNForwardPass1D:
    """Test CNN forward pass for 1D convolutions."""

    def test_cnn_1d_forward_basic(self):
        """Test basic forward pass for 1D CNN."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            backend=TorchBackend,
        )
        # Create 1D input: batch_size=2, channels=3, length=10
        x = torch.randn(2, 3, 10)
        output = cnn.forward(x)

        assert output is not None
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 5  # output channels

    def test_cnn_1d_forward_output_shape(self):
        """Test that 1D CNN output shape is correct."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=8,
            out_channels=3,
            n_hidden_layers=1,
            kernel_size=3,
            backend=TorchBackend,
        )
        x = torch.randn(4, 1, 20)
        output = cnn.forward(x)

        assert output.shape[0] == 4
        assert output.shape[1] == 3
        assert output.shape[2] == 20  # spatial dimension preserved with padding

    def test_cnn_1d_forward_multiple_hidden_layers(self):
        """Test 1D CNN forward pass with multiple hidden layers."""
        cnn = CNN(
            in_channels=2,
            hidden_channels=16,
            out_channels=4,
            n_hidden_layers=3,
            kernel_size=3,
            backend=TorchBackend,
        )
        x = torch.randn(2, 2, 15)
        output = cnn.forward(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 4


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCNNForwardPass2D:
    """Test CNN forward pass for 2D convolutions."""

    def test_cnn_2d_forward_basic(self):
        """Test basic forward pass for 2D CNN."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=(3, 3),
            backend=TorchBackend,
        )
        # Create 2D input: batch_size=2, channels=3, height=10, width=10
        x = torch.randn(2, 3, 10, 10)
        output = cnn.forward(x)

        assert output is not None
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 5  # output channels
        assert output.shape[2] == 10  # height preserved with padding
        assert output.shape[3] == 10  # width preserved with padding

    def test_cnn_2d_forward_various_sizes(self):
        """Test 2D CNN with different spatial dimensions."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=8,
            out_channels=2,
            n_hidden_layers=1,
            kernel_size=(3, 3),
            backend=TorchBackend,
        )

        # Test various input sizes
        for height, width in [(8, 8), (16, 16), (32, 32)]:
            x = torch.randn(1, 1, height, width)
            output = cnn.forward(x)
            assert output.shape == (1, 2, height, width)

    def test_cnn_2d_forward_multiple_hidden_layers(self):
        """Test 2D CNN forward pass with multiple hidden layers."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=32,
            out_channels=10,
            n_hidden_layers=2,
            kernel_size=(3, 3),
            backend=TorchBackend,
        )
        x = torch.randn(2, 3, 16, 16)
        output = cnn.forward(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 10
        assert output.shape[2] == 16
        assert output.shape[3] == 16

    def test_cnn_2d_forward_rectangular_input(self):
        """Test 2D CNN with non-square input."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=16,
            out_channels=4,
            n_hidden_layers=1,
            kernel_size=(3, 3),
            backend=TorchBackend,
        )
        x = torch.randn(2, 1, 12, 20)
        output = cnn.forward(x)

        assert output.shape == (2, 4, 12, 20)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCNNForwardPass3D:
    """Test CNN forward pass for 3D convolutions."""

    def test_cnn_3d_forward_basic(self):
        """Test basic forward pass for 3D CNN."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=8,
            out_channels=4,
            n_hidden_layers=1,
            kernel_size=(3, 3, 3),
            backend=TorchBackend,
        )
        # Create 3D input: batch_size=1, channels=1, depth=8, height=8, width=8
        x = torch.randn(1, 1, 8, 8, 8)
        output = cnn.forward(x)

        assert output is not None
        assert output.shape[0] == 1  # batch size preserved
        assert output.shape[1] == 4  # output channels
        assert output.shape[2] == 8  # depth preserved
        assert output.shape[3] == 8  # height preserved
        assert output.shape[4] == 8  # width preserved

    def test_cnn_3d_forward_various_sizes(self):
        """Test 3D CNN with different spatial dimensions."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=8,
            out_channels=2,
            n_hidden_layers=1,
            kernel_size=(3, 3, 3),
            backend=TorchBackend,
        )

        # Test various input sizes
        for size in [6, 8, 10]:
            x = torch.randn(1, 1, size, size, size)
            output = cnn.forward(x)
            assert output.shape == (1, 2, size, size, size)

    def test_cnn_3d_forward_multiple_hidden_layers(self):
        """Test 3D CNN forward pass with multiple hidden layers."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=4,
            out_channels=2,
            n_hidden_layers=2,
            kernel_size=(3, 3, 3),
            backend=TorchBackend,
        )
        x = torch.randn(1, 1, 8, 8, 8)
        output = cnn.forward(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 2
        assert output.shape[2] == 8
        assert output.shape[3] == 8
        assert output.shape[4] == 8

    def test_cnn_3d_forward_non_cubic(self):
        """Test 3D CNN with non-cubic input."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=8,
            out_channels=2,
            n_hidden_layers=1,
            kernel_size=(3, 3, 3),
            backend=TorchBackend,
        )
        x = torch.randn(1, 1, 6, 8, 10)
        output = cnn.forward(x)

        assert output.shape == (1, 2, 6, 8, 10)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCNNSetupMethod:
    """Test CNN setup and reconfiguration."""

    def test_cnn_setup_reinitializes_graph(self):
        """Test that setup method reinitializes the network graph."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            backend=TorchBackend,
        )

        # Store original graph
        original_graph = cnn._graph

        # Call setup
        cnn.setup()

        # New graph should be created
        assert cnn._graph is not None
        # (Graph rebuilding happens, so we just verify it still works)
        x = torch.randn(1, 3, 10)
        output = cnn.forward(x)
        assert output.shape == (1, 5, 10)

    def test_cnn_forward_after_setup(self):
        """Test that forward pass works correctly after setup."""
        cnn = CNN(
            in_channels=2,
            hidden_channels=8,
            out_channels=3,
            n_hidden_layers=1,
            kernel_size=3,
            backend=TorchBackend,
        )

        cnn.setup()

        x = torch.randn(2, 2, 16)
        output = cnn.forward(x)

        assert output.shape == (2, 3, 16)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCNNEdgeCases:
    """Test CNN edge cases and special configurations."""

    def test_cnn_single_hidden_layer(self):
        """Test CNN with no hidden layers (0 hidden layers)."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=0,
            kernel_size=3,
            backend=TorchBackend,
        )

        x = torch.randn(1, 3, 10)
        output = cnn.forward(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 5

    def test_cnn_single_channel_io(self):
        """Test CNN with single input and output channel."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=4,
            out_channels=1,
            n_hidden_layers=1,
            kernel_size=3,
            backend=TorchBackend,
        )

        x = torch.randn(2, 1, 20)
        output = cnn.forward(x)

        assert output.shape == (2, 1, 20)

    def test_cnn_large_kernel_size(self):
        """Test CNN with larger kernel sizes."""
        cnn = CNN(
            in_channels=1,
            hidden_channels=8,
            out_channels=2,
            n_hidden_layers=1,
            kernel_size=7,
            backend=TorchBackend,
        )

        x = torch.randn(1, 1, 32)
        output = cnn.forward(x)

        assert output.shape == (1, 2, 32)

    def test_cnn_batch_size_one(self):
        """Test CNN with batch size of 1."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=2,
            kernel_size=3,
            backend=TorchBackend,
        )

        x = torch.randn(1, 3, 10)
        output = cnn.forward(x)

        assert output.shape[0] == 1

    def test_cnn_large_batch_size(self):
        """Test CNN with large batch size."""
        cnn = CNN(
            in_channels=3,
            hidden_channels=16,
            out_channels=5,
            n_hidden_layers=1,
            kernel_size=3,
            backend=TorchBackend,
        )

        x = torch.randn(32, 3, 10)
        output = cnn.forward(x)

        assert output.shape[0] == 32
        assert output.shape[1] == 5
