import numpy as np
import pytest

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

from qewton.algorithms.building_blocks.activation_functions import ReLU, Tanh
from qewton.algorithms.dl_models.fcn import FCN, DeepRitzNet
from qewton.algorithms.dl_models.harmonic_fcn import HarmonicEmbedding, HarmonicFCN
from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.backends.numpy.base import NumPyBackend
from qewton.config.variables import Variable


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


DL_BACKENDS = all_subclasses(DeepLearningBackend)
COMPUTING_BACKENDS = [NumPyBackend, *DL_BACKENDS]


def build_input(backend: type[ComputingBackend], data):
    if backend is NumPyBackend:
        return np.asarray(data, dtype=np.float32)
    return backend.build_tensor(data)


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if HAS_TF and isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


class TestFCNInitialization:
    """Test FCN initialization with different configurations."""

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_fcn_basic_initialization(self, backend: type[DeepLearningBackend]):
        """Test basic FCN initialization."""
        fcn = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=1,
            backend=backend,
        )
        assert fcn is not None
        assert fcn.in_neurons.value == 3
        assert fcn.hidden_neurons.value == 16
        assert fcn.out_neurons.value == 5

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_fcn_variable_initialization(self, backend: type[DeepLearningBackend]):
        """Test FCN initialization with Variable inputs and outputs."""
        input_var = Variable("x", 2)
        output_var = Variable("u", 1)
        fcn = FCN(
            in_neurons=input_var,
            hidden_neurons=8,
            out_neurons=output_var,
            n_hidden_layers=2,
            backend=backend,
        )
        assert fcn.input_var == input_var
        assert fcn.output_var == output_var
        assert fcn.in_neurons.value == 2
        assert fcn.out_neurons.value == 1

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_fcn_multiple_hidden_layers(self, backend: type[DeepLearningBackend]):
        """Test FCN with multiple hidden layers."""
        fcn = FCN(
            in_neurons=4,
            hidden_neurons=12,
            out_neurons=2,
            n_hidden_layers=3,
            backend=backend,
        )
        assert fcn.n_hidden_layers.value == 3


class TestFCNHyperparameters:
    """Test FCN hyperparameter management."""

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_fcn_hyperparameters_property(self, backend: type[DeepLearningBackend]):
        """Test that hyperparameters property returns all hyperparameters."""
        fcn = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=2,
            backend=backend,
        )
        hps = fcn.hyperparameters
        assert len(hps) == 6

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_fcn_bias_hyperparameter(self, backend: type[DeepLearningBackend]):
        """Test FCN with and without bias."""
        fcn_with_bias = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=1,
            bias=True,
            backend=backend,
        )
        assert fcn_with_bias.bias.value is True

        fcn_without_bias = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=1,
            bias=False,
            backend=backend,
        )
        assert fcn_without_bias.bias.value is False

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_fcn_activation_function(self, backend: type[DeepLearningBackend]):
        """Test FCN with different activation functions."""
        fcn_relu = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=1,
            activation=ReLU,
            backend=backend,
        )
        assert fcn_relu.activation.value == ReLU

        fcn_tanh = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=1,
            activation=Tanh,
            backend=backend,
        )
        assert fcn_tanh.activation.value == Tanh


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestFCNForwardPass:
    """Test FCN forward pass behavior."""

    def test_fcn_forward_basic(self):
        """Test basic forward pass for FCN."""
        fcn = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=1,
            backend=TorchBackend,
        )
        x = torch.randn(2, 3)
        output = fcn.forward(x)

        assert output is not None
        assert output.shape == (2, 5)

    def test_fcn_forward_multiple_hidden_layers(self):
        """Test FCN forward pass with multiple hidden layers."""
        fcn = FCN(
            in_neurons=4,
            hidden_neurons=10,
            out_neurons=2,
            n_hidden_layers=3,
            backend=TorchBackend,
        )
        x = torch.randn(5, 4)
        output = fcn.forward(x)

        assert output.shape == (5, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestFCNSetupMethod:
    """Test FCN setup and reconfiguration."""

    def test_fcn_setup_reinitializes_graph(self):
        """Test that setup method reinitializes the network graph."""
        fcn = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=1,
            backend=TorchBackend,
        )

        original_graph = fcn._graph
        fcn.setup()

        assert fcn._graph is not None
        assert fcn._graph is not original_graph

        x = torch.randn(1, 3)
        output = fcn.forward(x)
        assert output.shape == (1, 5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestFCNEdgeCases:
    """Test FCN edge cases and special configurations."""

    def test_fcn_no_hidden_layers(self):
        """Test FCN with no hidden layers."""
        fcn = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=5,
            n_hidden_layers=0,
            backend=TorchBackend,
        )

        x = torch.randn(1, 3)
        output = fcn.forward(x)

        assert output.shape == (1, 5)

    def test_fcn_batch_size_one(self):
        """Test FCN with batch size of 1."""
        fcn = FCN(
            in_neurons=2,
            hidden_neurons=8,
            out_neurons=1,
            n_hidden_layers=2,
            backend=TorchBackend,
        )

        x = torch.randn(1, 2)
        output = fcn.forward(x)

        assert output.shape == (1, 1)

    def test_fcn_large_batch_size(self):
        """Test FCN with large batch size."""
        fcn = FCN(
            in_neurons=3,
            hidden_neurons=16,
            out_neurons=2,
            n_hidden_layers=1,
            backend=TorchBackend,
        )

        x = torch.randn(32, 3)
        output = fcn.forward(x)

        assert output.shape == (32, 2)


class TestHarmonicEmbeddingInitialization:
    """Test HarmonicEmbedding initialization with different configurations."""

    @pytest.mark.parametrize("backend", COMPUTING_BACKENDS)
    def test_harmonic_embedding_basic_initialization(
        self, backend: type[ComputingBackend]
    ):
        """Test basic harmonic embedding initialization."""
        embedding = HarmonicEmbedding(max_harmonic=3, backend=backend)
        assert embedding is not None
        assert embedding.max_harmonic.value == 3
        assert embedding.include_input.value is True

    @pytest.mark.parametrize("backend", COMPUTING_BACKENDS)
    def test_harmonic_embedding_without_input(self, backend: type[ComputingBackend]):
        """Test harmonic embedding initialization without input passthrough."""
        embedding = HarmonicEmbedding(
            max_harmonic=2,
            include_input=False,
            backend=backend,
        )
        assert embedding.include_input.value is False


class TestHarmonicEmbeddingHyperparameters:
    """Test HarmonicEmbedding hyperparameter management."""

    @pytest.mark.parametrize("backend", COMPUTING_BACKENDS)
    def test_harmonic_embedding_hyperparameters_property(
        self, backend: type[ComputingBackend]
    ):
        """Test that hyperparameters property returns all hyperparameters."""
        embedding = HarmonicEmbedding(max_harmonic=4, backend=backend)
        hps = embedding.hyperparameters
        assert len(hps) == 2


class TestHarmonicEmbeddingForwardPass:
    """Test HarmonicEmbedding forward pass behavior."""

    @pytest.mark.parametrize("backend", COMPUTING_BACKENDS)
    def test_harmonic_embedding_forward_with_input(self, backend: type[ComputingBackend]):
        """Test harmonic embedding forward pass including the original input."""
        embedding = HarmonicEmbedding(max_harmonic=3, include_input=True, backend=backend)
        x = build_input(backend, [[0.0], [0.25]])
        output = embedding.forward(x)

        assert tuple(output.shape) == (2, 7)

        expected = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.25, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(to_numpy(output), expected, atol=1e-5)

    @pytest.mark.parametrize("backend", COMPUTING_BACKENDS)
    def test_harmonic_embedding_forward_without_input(
        self, backend: type[ComputingBackend]
    ):
        """Test harmonic embedding forward pass without the original input."""
        embedding = HarmonicEmbedding(
            max_harmonic=2,
            include_input=False,
            backend=backend,
        )
        x = build_input(backend, [[0.0, 0.25]])
        output = embedding.forward(x)

        assert tuple(output.shape) == (1, 8)

        expected = np.array(
            [[1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(to_numpy(output), expected, atol=1e-5)


class TestHarmonicEmbeddingSetupMethod:
    """Test HarmonicEmbedding setup and reconfiguration."""

    def test_harmonic_embedding_setup_reinitializes_graph(self):
        """Test that setup method reinitializes the embedding graph."""
        embedding = HarmonicEmbedding(
            max_harmonic=3,
            include_input=True,
            backend=NumPyBackend,
        )

        original_graph = embedding._graph
        embedding.setup()

        assert embedding._graph is not None
        assert embedding._graph is not original_graph

        x = np.asarray([[0.5]], dtype=np.float32)
        output = embedding.forward(x)
        assert output.shape == (1, 7)


class TestDeepRitzNetInitialization:
    """Test DeepRitzNet initialization with different configurations."""

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_deep_ritz_net_basic_initialization(self, backend: type[DeepLearningBackend]):
        """Test basic DeepRitzNet initialization."""
        model = DeepRitzNet(
            in_neurons=3,
            out_neurons=1,
            width=16,
            depth=2,
            backend=backend,
        )
        assert model is not None
        assert model.in_neurons.value == 3
        assert model.hidden_neurons.value == 16
        assert model.out_neurons.value == 1
        assert model.n_hidden_layers.value == 2

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_deep_ritz_net_variable_initialization(
        self, backend: type[DeepLearningBackend]
    ):
        """Test DeepRitzNet initialization with Variable inputs and outputs."""
        input_var = Variable("x", 2)
        output_var = Variable("u", 1)
        model = DeepRitzNet(
            in_neurons=input_var,
            out_neurons=output_var,
            width=8,
            depth=3,
            backend=backend,
        )
        assert model.input_var == input_var
        assert model.output_var == output_var
        assert model.in_neurons.value == 2
        assert model.out_neurons.value == 1


class TestDeepRitzNetHyperparameters:
    """Test DeepRitzNet hyperparameter management."""

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_deep_ritz_net_hyperparameters_property(
        self, backend: type[DeepLearningBackend]
    ):
        """Test that hyperparameters property returns inherited hyperparameters."""
        model = DeepRitzNet(
            in_neurons=3,
            out_neurons=1,
            width=16,
            depth=2,
            backend=backend,
        )
        hps = model.hyperparameters
        assert len(hps) == 6


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDeepRitzNetForwardPass:
    """Test DeepRitzNet forward pass behavior."""

    def test_deep_ritz_net_forward_basic(self):
        """Test basic forward pass for DeepRitzNet."""
        model = DeepRitzNet(
            in_neurons=3,
            out_neurons=1,
            width=16,
            depth=2,
            backend=TorchBackend,
        )
        x = torch.randn(4, 3)
        output = model.forward(x)

        assert output is not None
        assert output.shape == (4, 1)

    def test_deep_ritz_net_forward_depth_zero(self):
        """Test DeepRitzNet with zero residual blocks."""
        model = DeepRitzNet(
            in_neurons=2,
            out_neurons=1,
            width=8,
            depth=0,
            backend=TorchBackend,
        )
        x = torch.randn(3, 2)
        output = model.forward(x)

        assert output.shape == (3, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDeepRitzNetSetupMethod:
    """Test DeepRitzNet setup and reconfiguration."""

    def test_deep_ritz_net_setup_reinitializes_graph(self):
        """Test that setup method reinitializes the network graph."""
        model = DeepRitzNet(
            in_neurons=3,
            out_neurons=1,
            width=16,
            depth=2,
            backend=TorchBackend,
        )

        original_graph = model._graph
        model.setup()

        assert model._graph is not None
        assert model._graph is not original_graph

        x = torch.randn(2, 3)
        output = model.forward(x)
        assert output.shape == (2, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDeepRitzNetEdgeCases:
    """Test DeepRitzNet edge cases and special configurations."""

    def test_deep_ritz_net_batch_size_one(self):
        """Test DeepRitzNet with batch size of 1."""
        model = DeepRitzNet(
            in_neurons=2,
            out_neurons=1,
            width=8,
            depth=1,
            backend=TorchBackend,
        )

        x = torch.randn(1, 2)
        output = model.forward(x)

        assert output.shape == (1, 1)

    def test_deep_ritz_net_large_batch_size(self):
        """Test DeepRitzNet with large batch size."""
        model = DeepRitzNet(
            in_neurons=3,
            out_neurons=2,
            width=12,
            depth=2,
            backend=TorchBackend,
        )

        x = torch.randn(32, 3)
        output = model.forward(x)

        assert output.shape == (32, 2)


class TestHarmonicFCNInitialization:
    """Test HarmonicFCN initialization with different configurations."""

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_harmonic_fcn_basic_initialization(self, backend: type[DeepLearningBackend]):
        """Test basic HarmonicFCN initialization."""
        model = HarmonicFCN(
            input_dim=2,
            hidden_neurons=16,
            output_dim=1,
            n_hidden_layers=2,
            max_harmonic=3,
            backend=backend,
        )
        assert model is not None
        assert model.embedding.max_harmonic.value == 3
        assert model.embedding.include_input.value is True
        assert model.fcn.hidden_neurons.value == 16
        assert model.fcn.n_hidden_layers.value == 2

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_harmonic_fcn_without_input_passthrough(
        self, backend: type[DeepLearningBackend]
    ):
        """Test HarmonicFCN with embedding input passthrough disabled."""
        model = HarmonicFCN(
            input_dim=3,
            hidden_neurons=8,
            output_dim=2,
            n_hidden_layers=1,
            max_harmonic=2,
            include_input=False,
            backend=backend,
        )
        assert model.embedding.include_input.value is False


class TestHarmonicFCNHyperparameters:
    """Test HarmonicFCN hyperparameter management."""

    @pytest.mark.parametrize("backend", DL_BACKENDS)
    def test_harmonic_fcn_hyperparameters_property(
        self, backend: type[DeepLearningBackend]
    ):
        """Test that hyperparameters include FCN and embedding hyperparameters."""
        model = HarmonicFCN(
            input_dim=2,
            hidden_neurons=8,
            output_dim=1,
            n_hidden_layers=1,
            max_harmonic=2,
            backend=backend,
        )
        hps = model.hyperparameters
        assert len(hps) == 8


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestHarmonicFCNForwardPass:
    """Test HarmonicFCN forward pass behavior."""

    def test_harmonic_fcn_forward_basic(self):
        """Test basic forward pass for HarmonicFCN."""
        model = HarmonicFCN(
            input_dim=2,
            hidden_neurons=16,
            output_dim=1,
            n_hidden_layers=1,
            max_harmonic=2,
            backend=TorchBackend,
        )
        x = torch.randn(4, 2)
        output = model.forward(x)

        assert output is not None
        assert output.shape == (4, 1)

    def test_harmonic_fcn_forward_without_input_passthrough(self):
        """Test HarmonicFCN forward pass with include_input disabled."""
        model = HarmonicFCN(
            input_dim=3,
            hidden_neurons=12,
            output_dim=2,
            n_hidden_layers=2,
            max_harmonic=3,
            include_input=False,
            backend=TorchBackend,
        )
        x = torch.randn(5, 3)
        output = model.forward(x)

        assert output.shape == (5, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestHarmonicFCNSetupMethod:
    """Test HarmonicFCN setup and reconfiguration."""

    def test_harmonic_fcn_setup_reinitializes_subgraphs(self):
        """Test that setup rebuilds embedding and FCN subgraphs."""
        model = HarmonicFCN(
            input_dim=2,
            hidden_neurons=8,
            output_dim=1,
            n_hidden_layers=1,
            max_harmonic=2,
            backend=TorchBackend,
        )

        original_embedding_graph = model.embedding._graph
        original_fcn_graph = model.fcn._graph
        model.setup()

        assert model.embedding._graph is not original_embedding_graph
        assert model.fcn._graph is original_fcn_graph

        x = torch.randn(2, 2)
        output = model.forward(x)
        assert output.shape == (2, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestHarmonicFCNEdgeCases:
    """Test HarmonicFCN edge cases and special configurations."""

    def test_harmonic_fcn_batch_size_one(self):
        """Test HarmonicFCN with batch size of 1."""
        model = HarmonicFCN(
            input_dim=1,
            hidden_neurons=8,
            output_dim=1,
            n_hidden_layers=1,
            max_harmonic=4,
            backend=TorchBackend,
        )

        x = torch.randn(1, 1)
        output = model.forward(x)

        assert output.shape == (1, 1)

    def test_harmonic_fcn_large_batch_size(self):
        """Test HarmonicFCN with large batch size."""
        model = HarmonicFCN(
            input_dim=2,
            hidden_neurons=16,
            output_dim=3,
            n_hidden_layers=2,
            max_harmonic=2,
            backend=TorchBackend,
        )

        x = torch.randn(32, 2)
        output = model.forward(x)

        assert output.shape == (32, 3)
