import pytest

try:
    import torch
    from qewton.backends.torch.base import TorchBackend

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

from qewton.algorithms.dl_models.convolutions.encoding import ConvolutionalEncoder
from qewton.algorithms.building_blocks.activation_functions import ReLU, Tanh
from qewton.backends.base import DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.config.variables import Variable


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(DeepLearningBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


class TestConvolutionalEncoderInitialization:
    """Test encoder initialization with different configurations."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_encoder_1d_initialization(self, backend: type[DeepLearningBackend]):
        encoder = ConvolutionalEncoder(
            in_channels=3,
            channels=(8, 16),
            out_channels=5,
            conv_kernel_size=(3,),
            input_shape=(3, 32),
            backend=backend,
        )
        assert encoder is not None
        assert encoder.in_channels.value == 3
        assert encoder.out_channels.value == 5
        assert encoder.image_dim == 1

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_encoder_2d_initialization(self, backend: type[DeepLearningBackend]):
        encoder = ConvolutionalEncoder(
            in_channels=1,
            channels=(8, 16),
            out_channels=10,
            conv_kernel_size=(3, 3),
            input_shape=(1, 16, 16),
            backend=backend,
        )
        assert encoder is not None
        assert encoder.image_dim == 2
        assert len(encoder.channels) == 2

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_encoder_3d_initialization(self, backend: type[DeepLearningBackend]):
        encoder = ConvolutionalEncoder(
            in_channels=1,
            channels=(4, 8),
            out_channels=6,
            conv_kernel_size=(3, 3, 3),
            input_shape=(1, 8, 8, 8),
            backend=backend,
        )
        assert encoder is not None
        assert encoder.image_dim == 3

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_encoder_variable_channels(self, backend: type[DeepLearningBackend]):
        in_var = Variable("u", 2)
        out_var = Variable("v", 7)
        encoder = ConvolutionalEncoder(
            in_channels=in_var,
            channels=(8,),
            out_channels=out_var,
            conv_kernel_size=(3,),
            input_shape=(2, 20),
            backend=backend,
        )
        assert encoder.input_var == in_var
        assert encoder.output_var == out_var
        assert encoder.in_channels.value == 2
        assert encoder.out_channels.value == 7


class TestConvolutionalEncoderHyperparameters:
    """Test encoder hyperparameter registration."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_encoder_activation_hyperparameter(self, backend: type[DeepLearningBackend]):
        encoder_relu = ConvolutionalEncoder(
            in_channels=1,
            channels=(8,),
            out_channels=3,
            conv_kernel_size=(3,),
            activation=ReLU,
            input_shape=(1, 16),
            backend=backend,
        )
        assert encoder_relu.activation.value == ReLU

        encoder_tanh = ConvolutionalEncoder(
            in_channels=1,
            channels=(8,),
            out_channels=3,
            conv_kernel_size=(3,),
            activation=Tanh,
            input_shape=(1, 16),
            backend=backend,
        )
        assert encoder_tanh.activation.value == Tanh


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestConvolutionalEncoderForwardPass:
    """Test encoder forward pass for different dimensions."""

    def test_encoder_1d_forward(self):
        encoder = ConvolutionalEncoder(
            in_channels=3,
            channels=(8, 16),
            out_channels=5,
            conv_kernel_size=(3,),
            pooling_kernel_size=2,
            input_shape=(3, 32),
            backend=TorchBackend,
        )
        x = torch.randn(4, 3, 32)
        y = encoder.forward(x)
        assert y.shape == (4, 5)

    def test_encoder_2d_forward(self):
        encoder = ConvolutionalEncoder(
            in_channels=1,
            channels=(8, 16),
            out_channels=6,
            conv_kernel_size=(3, 3),
            pooling_kernel_size=(2, 2),
            input_shape=(1, 16, 16),
            backend=TorchBackend,
        )
        x = torch.randn(2, 1, 16, 16)
        y = encoder.forward(x)
        assert y.shape == (2, 6)

    def test_encoder_3d_forward(self):
        encoder = ConvolutionalEncoder(
            in_channels=1,
            channels=(4, 8),
            out_channels=3,
            conv_kernel_size=(3, 3, 3),
            pooling_kernel_size=(2, 2, 2),
            input_shape=(1, 8, 8, 8),
            backend=TorchBackend,
        )
        x = torch.randn(1, 1, 8, 8, 8)
        y = encoder.forward(x)
        assert y.shape == (1, 3)

    def test_encoder_forward_infers_input_shape(self):
        encoder = ConvolutionalEncoder(
            in_channels=2,
            channels=(8,),
            out_channels=4,
            conv_kernel_size=(3,),
            input_shape=None,
            backend=TorchBackend,
        )
        x = torch.randn(3, 2, 20)
        y = encoder.forward(x)
        assert encoder.input_shape == (2, 20)
        assert y.shape == (3, 4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestConvolutionalEncoderSetupMethod:
    """Test setup and reset behavior."""

    def test_encoder_setup_reinitializes_graph(self):
        encoder = ConvolutionalEncoder(
            in_channels=2,
            channels=(8, 16),
            out_channels=5,
            conv_kernel_size=(3,),
            input_shape=(2, 24),
            backend=TorchBackend,
        )
        old_graph = encoder._graph
        encoder.setup()
        assert encoder._graph is not old_graph
        x = torch.randn(2, 2, 24)
        y = encoder.forward(x)
        assert y.shape == (2, 5)

    def test_encoder_reset_sets_uninitialized(self):
        encoder = ConvolutionalEncoder(
            in_channels=1,
            channels=(8,),
            out_channels=3,
            conv_kernel_size=(3,),
            input_shape=(1, 16),
            backend=TorchBackend,
        )
        encoder.setup()
        encoder.reset()
        x = torch.randn(1, 1, 16)
        y = encoder.forward(x)
        assert y.shape == (1, 3)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestConvolutionalEncoderEdgeCases:
    """Test edge case configurations."""

    def test_encoder_max_pooling(self):
        encoder = ConvolutionalEncoder(
            in_channels=1,
            channels=(8,),
            out_channels=2,
            conv_kernel_size=(3, 3),
            pooling_type="maximum",
            pooling_kernel_size=(2, 2),
            input_shape=(1, 12, 12),
            backend=TorchBackend,
        )
        x = torch.randn(1, 1, 12, 12)
        y = encoder.forward(x)
        assert y.shape == (1, 2)

    def test_encoder_large_batch(self):
        encoder = ConvolutionalEncoder(
            in_channels=3,
            channels=(8, 16),
            out_channels=4,
            conv_kernel_size=(3,),
            input_shape=(3, 32),
            backend=TorchBackend,
        )
        x = torch.randn(16, 3, 32)
        y = encoder.forward(x)
        assert y.shape == (16, 4)
