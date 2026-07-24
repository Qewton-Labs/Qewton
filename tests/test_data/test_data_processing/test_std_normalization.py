import inspect
import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.data.dataloaders.base import DataNode
from qewton.data.data_processing.normalization import (
    StdNormalizationNode,
    InverseStdNormalizationNode,
)
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
    """Mock DataNode that provides dummy data for testing normalization."""

    def __init__(self, data_batches, n_batches=3, batch_size=10, backend=None):
        super().__init__(batch_size=batch_size, name="DummyDataNode", backend=backend)
        self.data_batches = data_batches
        self.n_batches = n_batches
        self._batch_idx = 0
        self._state = NodeState.FIXED

        config = DataConfiguration(BatchAxes(AxesDim(batch_size)), EllipsisAxes())
        self._output_ports = [OutputPort(config, self, name="output")]

    def __len__(self):
        return self.n_batches

    def forward(self):
        if self._batch_idx >= len(self.data_batches):
            self._batch_idx = 0
        batch = self.data_batches[self._batch_idx]
        self._batch_idx += 1
        return batch

    def provides_data_in_phase(self, phase):
        return True


@pytest.mark.parametrize("backend", BACKENDS)
def test_std_normalization_initialization(backend):
    dummy = [
        backend.build_tensor([[float(i + j * 0.1) for j in range(4)] for i in range(6)])
    ]
    data_node = DummyDataNode(dummy, n_batches=1, batch_size=6, backend=backend)

    norm = StdNormalizationNode(data_source_node=data_node, eps=1e-6, backend=backend)

    assert norm.eps == 1e-6
    assert norm.state == NodeState.UNINITIALIZED


@pytest.mark.parametrize("backend", BACKENDS)
def test_std_normalization_setup_computes_mean_std(backend):
    # create three batches with easily computed means
    b1 = backend.build_tensor([[0.0, 1.0, 2.0, 3.0] for _ in range(5)])
    b2 = backend.build_tensor([[1.0, 2.0, 3.0, 4.0] for _ in range(5)])
    b3 = backend.build_tensor([[2.0, 3.0, 4.0, 5.0] for _ in range(5)])
    data_node = DummyDataNode([b1, b2, b3], n_batches=3, batch_size=5, backend=backend)

    norm = StdNormalizationNode(data_source_node=data_node, backend=backend)
    graph = Graph()
    graph.connect(data_node, norm)
    graph.setup()

    assert norm.state == NodeState.INITIALIZED
    # mean should be mean of concatenated batches: values 0..2 averaged -> 1.0,2.0,3.0,4.0
    expected_mean = backend.build_tensor([[1.0, 2.0, 3.0, 4.0]])
    assert backend.math.allclose(norm.mean, expected_mean)
    # std should be > 0
    assert backend.math.all(norm.std > 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_std_and_inverse_roundtrip(backend):
    # Use a small dataset and check that inverse restores original values
    b = backend.build_tensor([[float(i + j * 0.1) for j in range(4)] for i in range(6)])
    data_node = DummyDataNode([b], n_batches=1, batch_size=6, backend=backend)

    norm = StdNormalizationNode(data_source_node=data_node, backend=backend)
    inv = InverseStdNormalizationNode(std_node=norm)

    graph = Graph()
    graph.connect(data_node, norm)
    graph.connect(norm, inv)
    graph.setup()

    # pick some input to normalize and invert
    x = backend.build_tensor([[float(i + j * 0.1) for j in range(4)] for i in range(3)])
    normalized = norm.forward(x)
    reconstructed = inv.forward(normalized)

    assert reconstructed.shape == x.shape
    assert backend.math.allclose(reconstructed, x, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_std_normalization_device_transfer(backend, device):
    if not issubclass(backend, DeepLearningBackend):
        pytest.skip("Device transfer only relevant for DeepLearningBackend")

    b = backend.build_tensor([[float(i + j * 0.1) for j in range(4)] for i in range(6)])
    data_node = DummyDataNode([b], n_batches=1, batch_size=6, backend=backend)
    norm = StdNormalizationNode(data_source_node=data_node, backend=backend)
    inv = InverseStdNormalizationNode(std_node=norm)

    graph = Graph()
    graph.connect(data_node, norm)
    graph.connect(norm, inv)
    graph.setup()

    norm.to(device=device)
    inv.to(device=device)
    x = backend.build_tensor([[float(i + j * 0.1) for j in range(4)] for i in range(2)])
    x = backend.to(x, device=device)

    normalized = norm.forward(x)
    reconstructed = inv.forward(normalized)

    assert reconstructed.shape == x.shape
