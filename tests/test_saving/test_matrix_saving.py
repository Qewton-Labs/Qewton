"""Test matrix of saving models: model × check (config V1).

Every built-in model is built (sizes inferred from data where they are
open), saved and loaded. The loaded model computes the same as the saved
one; the static configurations of all its ports keep their dtypes; the
dims of its ports are bound to its own HyperParameters, so tuning the
loaded model changes them; inferred sizes are kept. Cases that do not hold
yet are listed in GAPS with their reason.
"""

from collections import Counter
from pathlib import Path
from typing import Callable

import pytest
import torch

from qewton.algorithms.building_blocks import ReLU
from qewton.algorithms.dl_models.convolutions.cnn import CNN, UNet
from qewton.algorithms.dl_models.convolutions.encoding import ConvolutionalEncoder
from qewton.algorithms.dl_models.deeponet.deeponet_cnn import CNNDeepONet
from qewton.algorithms.dl_models.deeponet.deeponet_fcn import FCNDeepONet
from qewton.algorithms.dl_models.fcn import FCN, DeepRitzNet
from qewton.algorithms.dl_models.fno import FNO
from qewton.algorithms.dl_models.harmonic_fcn import HarmonicFCN
from qewton.algorithms.dl_models.pca_net import PCANet
from qewton.config.data_configurations import DataConfiguration
from qewton.config.devices import cpu, cuda
from qewton.config.saving.callables import load, save
from qewton.config.variables import Variable
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node, OutputPort
from qewton.optim.parameters.helpers import HyperParameterState
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class Values(Node):
    """A node providing a fixed value."""

    def __init__(self, value):
        super().__init__(name="values")
        self.value = value
        self._output_ports = [
            OutputPort(DataConfiguration.of_value(value), node=self, name="out")
        ]

    def forward(self):
        return self.value


def _connected(model: Node, *inputs) -> Node:
    """`model` set up in a graph fed with `inputs`, so that its open sizes
    are inferred and it is built."""
    graph = Graph()
    for value, port in zip(inputs, model.input_ports):
        graph.connect(Values(value), port)
    graph.setup()
    return model


def _set_up(model: Node) -> Node:
    model.setup()
    return model


def _pca_net():
    model = PCANet(
        input_variable=Variable("x", dim=3), output_variable=Variable("o", dim=1),
        pca_n_input=3, pca_n_output=1, data_source_node=None,  # type: ignore
        fcn_hidden_layers=1, fcn_hidden_neurons=4,
    )
    torch.manual_seed(0)
    model.fit([torch.rand(100, 3)], [torch.rand(100, 1)])
    return model


#: name -> (built model, inputs, the input size it is bound to or None)
MODELS: dict[str, tuple[Callable[[], Node], Callable[[], tuple], Callable | None]] = {
    "fcn": (
        lambda: FCN(in_neurons=2, hidden_neurons=4, out_neurons=1, n_hidden_layers=3),
        lambda: (torch.rand(10, 2),),
        lambda m: m.in_neurons,
    ),
    "fcn_without_bias": (
        lambda: FCN(in_neurons=2, hidden_neurons=25, out_neurons=2, n_hidden_layers=5,
                    bias=False),
        lambda: (torch.rand(2, 2),),
        lambda m: m.in_neurons,
    ),
    "fcn_inferred": (
        lambda: _connected(
            FCN(in_neurons=None, hidden_neurons=4, out_neurons=1, n_hidden_layers=1),
            torch.rand(10, 3),
        ),
        lambda: (torch.rand(10, 3),),
        lambda m: m.in_neurons,
    ),
    "deep_ritz": (
        lambda: DeepRitzNet(in_neurons=2, width=10, out_neurons=1, depth=1),
        lambda: (torch.rand(2, 2),),
        None,
    ),
    "harmonic_fcn": (
        lambda: _set_up(HarmonicFCN(
            input_dim=2, hidden_neurons=4, output_dim=1, n_hidden_layers=3,
            max_harmonic=3, bias=True, activation=ReLU,
        )),
        lambda: (torch.rand(2, 2),),
        None,
    ),
    "cnn": (
        lambda: _set_up(CNN(in_channels=2, hidden_channels=16, out_channels=1,
                            n_hidden_layers=3, kernel_size=(3, 3))),
        lambda: (torch.rand(1, 2, 5, 5),),
        lambda m: m.in_channels,
    ),
    "unet": (
        lambda: _set_up(UNet(in_channels=2, channels=(16, 32, 64), out_channels=1,
                             conv_kernel_size=(3, 3))),
        lambda: (torch.rand(1, 2, 20, 20),),
        lambda m: m.in_channels,
    ),
    "encoding": (
        lambda: _set_up(ConvolutionalEncoder(
            in_channels=2, channels=(16, 16), out_channels=1, conv_kernel_size=(3, 3),
            input_shape=(32, 32),
        )),
        lambda: (torch.rand(3, 2, 32, 32),),
        lambda m: m.in_channels,
    ),
    "pca_net": (_pca_net, lambda: (torch.rand(100, 3),), None),
    "fcn_deeponet": (
        lambda: _connected(
            FCNDeepONet(trunk_input=2, branch_input=1, output=3, trunk_hidden_neurons=4,
                        branch_hidden_neurons=4, trunk_hidden_layers=1,
                        branch_hidden_layers=1, intermediate_neurons=3),
            torch.rand(4, 6, 1), torch.rand(4, 5, 2),
        ),
        lambda: (torch.rand(4, 6, 1), torch.rand(4, 5, 2)),
        None,
    ),
    "cnn_deeponet": (
        lambda: _connected(
            CNNDeepONet(trunk_input=2, branch_input=1, output=3, trunk_hidden_neurons=4,
                        trunk_hidden_layers=1, channels=(4,), conv_kernel_size=(3,),
                        intermediate_neurons=3, branch_fcn_hidden_neurons=4,
                        branch_fcn_hidden_layers=1),
            torch.rand(4, 1, 16), torch.rand(4, 5, 2),
        ),
        lambda: (torch.rand(4, 1, 16), torch.rand(4, 5, 2)),
        None,
    ),
    "fno": (
        lambda: FNO(n_modes=(4,), in_channels=2, out_channels=1, hidden_channels=4,
                    n_layers=1),
        lambda: (torch.rand(3, 2, 16),),
        lambda m: m.in_channels,
    ),
}


def _ports(model: Node) -> list:
    """Every port of `model` and of the nodes inside it."""
    ports = list(model.input_ports) + list(model.output_ports)
    if isinstance(model, GraphNode):
        for node in model.inner_graph.nodes:
            ports += _ports(node)
    return ports


def _dtypes(model: Node) -> Counter:
    return Counter(
        (type(p.node).__name__, p.name, str(p.data_configuration.dtype),
         str(p.data_configuration.element_dtype))
        for p in _ports(model)
    )


def _inferred(model: Node) -> list:
    return [hp.value for hp in model.hyperparameters
            if hp.state == HyperParameterState.INFERRED]


def check_computes_the_same(model, loaded, name):
    inputs = MODELS[name][1]()
    torch.manual_seed(0)
    original = model(*inputs)
    torch.manual_seed(0)
    assert torch.allclose(original, loaded(*inputs))


def check_keeps_the_dtypes(model, loaded, name):
    assert _dtypes(loaded) == _dtypes(model)


def check_sizes_are_bound_to_its_hyperparameters(model, loaded, name):
    size_of = MODELS[name][2]
    if size_of is None:
        pytest.skip("no input size of its own on its ports")
    size = size_of(loaded)
    feature_dim = loaded.input_ports[0].data_configuration.feature_axes.shape[0]
    size.set_value(size.value + 1)
    assert feature_dim.size == size.value


def check_keeps_inferred_sizes(model, loaded, name):
    if not _inferred(model):
        pytest.skip("no inferred size")
    assert _inferred(loaded) == _inferred(model)


CHECKS = {
    "computes_the_same": check_computes_the_same,
    "keeps_the_dtypes": check_keeps_the_dtypes,
    "sizes_are_bound_to_its_hyperparameters": check_sizes_are_bound_to_its_hyperparameters,
    "keeps_inferred_sizes": check_keeps_inferred_sizes,
}


GAPS: dict[tuple[str | None, str], str] = {}


def _cases():
    for name in MODELS:
        for check in CHECKS:
            reason = GAPS.get((name, check)) or GAPS.get((None, check))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(name, check, marks=marks, id=f"{name}-{check}")


@pytest.mark.parametrize("name, check", _cases())
def test_saving(name, check, tmp_path):
    model = MODELS[name][0]()
    save(model, tmp_path / name, replace=True)
    loaded = load(tmp_path / name)
    assert type(loaded) is type(model)
    CHECKS[check](model, loaded, name)


#: Models saved by qewton before structured dtypes (format version 1), see
#: `version_1/make_files.py`, with their input and output.
VERSION_1 = Path(__file__).parent / "version_1"


@pytest.mark.parametrize("name", sorted(p.name for p in VERSION_1.iterdir() if p.is_dir()))
def test_a_model_of_format_version_1_computes_as_saved(name):
    loaded = load(VERSION_1 / name)
    expected = torch.load(VERSION_1 / name / "expected.pt")
    assert torch.allclose(loaded(expected["input"]), expected["output"])


def test_a_file_of_an_unknown_format_version_is_refused(tmp_path):
    save(FCN(in_neurons=2, hidden_neurons=2, out_neurons=1, n_hidden_layers=1), tmp_path / "m")
    config = tmp_path / "m" / "config.json"
    config.write_text(config.read_text().replace('"version": 2', '"version": 99'))
    with pytest.raises(ValueError, match="format version 99"):
        load(tmp_path / "m")


def test_an_open_size_stays_open(tmp_path):
    model = FCN(in_neurons=None, hidden_neurons=4, out_neurons=1, n_hidden_layers=1)
    save(model, tmp_path / "open", replace=True)
    loaded = load(tmp_path / "open")
    assert [hp.name for hp in loaded.open_hyperparameters] == ["FCN Input Neurons"]


@pytest.mark.parametrize("device", [cpu, cuda(1)], ids=["cpu", "cuda_1"])
def test_a_device_is_loaded_as_the_same_object(device, tmp_path):
    save(HyperParameter.from_value(device, "device"), tmp_path / "device", replace=True)
    assert load(tmp_path / "device").value is device
