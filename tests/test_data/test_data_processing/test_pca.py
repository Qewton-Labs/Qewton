import inspect
import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.data.dataloaders.base import DataNode
from qewton.data.data_processing.pca import PCANode, InversePCANode
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import OutputPort, NodeState
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import BatchAxes, AxesDim, EllipsisAxes


def all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


class DummyDataNode(DataNode):
    """Mock DataNode that provides dummy data for testing PCA."""

    def __init__(self, data_batches, n_batches=3, batch_size=10, backend=None):
        """
        Args:
            data_batches: List of data tensors, one per batch
            n_batches: Number of batches (training_batches)
            batch_size: Size of each batch
            backend: The backend to use
        """
        super().__init__(batch_size=batch_size, name="DummyDataNode", backend=backend)
        self.data_batches = data_batches
        self.n_batches = n_batches
        self._batch_idx = 0
        self._state = NodeState.FIXED

        # Create output port with data
        config = DataConfiguration(BatchAxes(AxesDim(batch_size)), EllipsisAxes())
        self._output_ports = [OutputPort(config, self, name="output")]

    def __len__(self):
        return self.n_batches

    def forward(self):
        """Return next batch of data."""
        if self._batch_idx >= len(self.data_batches):
            self._batch_idx = 0
        batch = self.data_batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def provides_data_in_phase(self, phase):
        return True


@pytest.mark.parametrize("backend", BACKENDS)
def test_pca_node_initialization(backend):
    """Test PCANode initialization with different parameters."""
    # Create dummy data
    dummy_data = [
        backend.build_tensor([[i + j * 0.1 for j in range(5)] for i in range(10)])
    ]
    data_node = DummyDataNode(dummy_data, n_batches=1, batch_size=10, backend=backend)

    pca_node = PCANode(n=3, data_source_node=data_node, scale=True, backend=backend)

    assert pca_node.n.current_value == 3
    assert pca_node.scale.current_value is True
    assert pca_node.state == NodeState.UNINITIALIZED


@pytest.mark.parametrize("backend", BACKENDS)
def test_pca_node_setup_collects_data(backend):
    """Test that PCANode.setup() collects all data and computes PCA."""
    # Create dummy data - 3 batches
    dummy_data = [
        backend.build_tensor([[float(i + j * 0.1) for j in range(8)] for i in range(10)]),
        backend.build_tensor(
            [[float(i + 1 + j * 0.1) for j in range(8)] for i in range(10)]
        ),
        backend.build_tensor(
            [[float(i + 2 + j * 0.1) for j in range(8)] for i in range(10)]
        ),
    ]
    data_node = DummyDataNode(dummy_data, n_batches=3, batch_size=10, backend=backend)

    pca_node = PCANode(n=3, data_source_node=data_node, scale=True, backend=backend)

    # Create graph and setup
    graph = Graph()
    graph.connect(data_node, pca_node)
    graph.setup()

    assert pca_node.state == NodeState.INITIALIZED
    assert pca_node.pca_u is not None
    assert pca_node.pca_s is not None
    assert pca_node.pca_v is not None


@pytest.mark.parametrize("backend", BACKENDS)
def test_pca_node_output_shapes(backend):
    """Test that PCANode produces correct output shapes."""
    n_samples = 10
    n_features = 8
    n_components = 3

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=True, backend=backend
    )
    graph = Graph()
    graph.connect(data_node, pca_node)
    graph.setup()

    # Test output shapes from forward
    x = backend.build_tensor(
        [[float(i + j * 0.1) for j in range(n_features)] for i in range(5)]
    )
    output, u, s, v = pca_node.forward(x)

    assert output.shape == (
        5,
        n_components,
    ), f"Expected (5, {n_components}), got {output.shape}"
    assert u.shape[1] == n_components
    assert s.shape[0] == n_components
    assert v.shape[1] == n_components


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_pca_node_forward_evaluation(backend, device):
    """Test PCANode forward pass on different devices."""
    n_samples = 10
    n_features = 6
    n_components = 2

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=False, backend=backend
    )
    graph = Graph()
    graph.connect(data_node, pca_node)
    graph.setup()

    if issubclass(backend, DeepLearningBackend):
        pca_node.to(device=device)

    x = backend.build_tensor(
        [[float(i + j * 0.1) for j in range(n_features)] for i in range(3)]
    )
    if issubclass(backend, DeepLearningBackend):
        x = backend.to(x, device=device)

    output, _, _, _ = pca_node.forward(x)

    assert output.shape == (3, n_components)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pca_node_with_scaling(backend):
    """Test PCANode with and without scaling."""
    n_samples = 10
    n_features = 5
    n_components = 2

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    # With scaling
    pca_node_scaled = PCANode(
        n=n_components, data_source_node=data_node, scale=True, backend=backend
    )
    graph = Graph()

    x = backend.build_tensor(
        [[float(i + j * 0.1) for j in range(n_features)] for i in range(3)]
    )

    # Without scaling
    pca_node_unscaled = PCANode(
        n=n_components, data_source_node=data_node, scale=False, backend=backend
    )
    graph.connect(data_node, pca_node_scaled)
    graph.connect(data_node, pca_node_unscaled)
    graph.setup()
    output_unscaled, _, _, _ = pca_node_unscaled.forward(x)
    output_scaled, _, _, _ = pca_node_scaled.forward(x)
    # Outputs should be different due to scaling
    assert not backend.math.allclose(output_scaled, output_unscaled)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inverse_pca_node_initialization(backend):
    """Test InversePCANode initialization."""
    n_samples = 10
    n_features = 5
    n_components = 2

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=True, backend=backend
    )
    inverse_pca = InversePCANode(pca_node=pca_node)

    assert inverse_pca.data_source_node is pca_node
    assert inverse_pca.state == NodeState.UNINITIALIZED


@pytest.mark.parametrize("backend", BACKENDS)
def test_inverse_pca_node_setup_requires_initialized_pca(backend):
    """Test that InversePCANode.setup() fails if PCA is not initialized."""
    n_samples = 10
    n_features = 5
    n_components = 2

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=True, backend=backend
    )
    inverse_pca = InversePCANode(pca_node=pca_node)

    graph = Graph()
    with pytest.raises(RuntimeError):
        inverse_pca.setup(graph)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inverse_pca_node_forward_reconstruction(backend):
    """Test InversePCANode forward pass reconstructs data."""
    n_samples = 10
    n_features = 6
    n_components = 3

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=False, backend=backend
    )
    inverse_pca = InversePCANode(pca_node=pca_node)
    graph = Graph()
    graph.connect(data_node, pca_node)
    graph.connect(pca_node.output, inverse_pca)
    graph.setup()

    x = backend.build_tensor(
        [[float(i + j * 0.1) for j in range(n_features)] for i in range(5)]
    )

    # Forward through PCA
    pca_coeffs, _, _, _ = pca_node.forward(x)

    # Backward through inverse PCA
    reconstructed = inverse_pca.forward(pca_coeffs)

    # Should have same shape as original (flattened)
    assert reconstructed.shape[0] == x.shape[0]
    assert reconstructed.shape[1] == n_features


@pytest.mark.parametrize("backend", BACKENDS)
def test_pca_roundtrip_reconstruction(backend):
    """Test that PCA -> InversePCA approximately recovers original data."""
    n_samples = 10
    n_features = 8
    n_components = 5

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=False, backend=backend
    )
    inverse_pca = InversePCANode(pca_node=pca_node)
    graph = Graph()
    graph.connect(data_node, pca_node)
    graph.connect(pca_node.output, inverse_pca)
    graph.setup()
    x = backend.build_tensor(
        [[float(i + j * 0.1) for j in range(n_features)] for i in range(4)]
    )

    # Apply PCA then inverse
    pca_coeffs, _, _, _ = pca_node.forward(x)
    reconstructed = inverse_pca.forward(pca_coeffs)

    # For full rank PCA (n_components == n_features), reconstruction should be very close
    if n_components == n_features:
        assert backend.math.allclose(reconstructed, x, rtol=1e-5)
    else:
        # For reduced rank, check that at least some correlation exists
        # (reconstruction should not be completely random)
        correlation = backend.math.sum(reconstructed * x) / backend.math.sqrt(
            backend.math.sum(reconstructed**2) * backend.math.sum(x**2)
        )
        assert backend.math.abs(correlation) > 0.4


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_pca_and_inverse_pca_device_transfer(backend, device):
    """Test device transfer for both PCA and InversePCA nodes."""
    if not issubclass(backend, DeepLearningBackend):
        pytest.skip("Device transfer only relevant for DeepLearningBackend")

    n_samples = 10
    n_features = 5
    n_components = 2

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=True, backend=backend
    )
    inverse_pca = InversePCANode(pca_node=pca_node)
    graph = Graph()
    graph.connect(data_node, pca_node)
    graph.connect(pca_node.output, inverse_pca)
    graph.setup()

    pca_node.to(device=device)

    x = backend.build_tensor(
        [[float(i + j * 0.1) for j in range(n_features)] for i in range(3)]
    )
    x = backend.to(x, device=device)

    pca_coeffs, _, _, _ = pca_node.forward(x)
    _ = inverse_pca.forward(pca_coeffs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pca_multiple_batches_collection(backend):
    """Test that PCANode correctly collects and uses data from multiple batches."""
    n_samples_per_batch = 5
    n_features = 4
    n_components = 2
    n_batches = 4

    # Create 4 batches with different data patterns
    dummy_data = [
        backend.build_tensor(
            [
                [float(i + b + j * 0.1) for j in range(n_features)]
                for i in range(n_samples_per_batch)
            ]
        )
        for b in range(n_batches)
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=n_batches, batch_size=n_samples_per_batch, backend=backend
    )

    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=True, backend=backend
    )
    graph = Graph()
    graph.connect(data_node, pca_node)
    graph.setup()

    # Check that pca_u has correct shape (should reflect collected data dimension)
    assert pca_node.pca_u.shape[1] == n_components
    assert pca_node.pca_s.shape[0] == n_components
    assert pca_node.pca_v.shape[1] == n_components


@pytest.mark.parametrize("backend", BACKENDS)
def test_inverse_pca_with_scaling(backend):
    """Test InversePCANode correctly unscales when PCA was scaled."""
    n_samples = 10
    n_features = 6
    n_components = 3

    dummy_data = [
        backend.build_tensor(
            [[float(i + j * 0.1) for j in range(n_features)] for i in range(n_samples)]
        )
    ]
    data_node = DummyDataNode(
        dummy_data, n_batches=1, batch_size=n_samples, backend=backend
    )

    # Create PCA with scaling
    pca_node = PCANode(
        n=n_components, data_source_node=data_node, scale=True, backend=backend
    )
    graph = Graph()
    inverse_pca = InversePCANode(pca_node=pca_node)
    graph.connect(data_node, pca_node)
    graph.connect(pca_node.output, inverse_pca)
    graph.setup()
    x = backend.build_tensor(
        [[float(i + j * 0.1) for j in range(n_features)] for i in range(5)]
    )

    # Apply PCA then inverse
    pca_coeffs, _, _, _ = pca_node.forward(x)
    reconstructed = inverse_pca.forward(pca_coeffs)

    # Should have reconstructed shape
    assert reconstructed.shape == (5, n_features)
