"""Test matrix of sizes that fix how a model is built (config H1–H3): model × origin
× places.

Every model has an input size that is a HyperParameter (`SIZES`). Its
origin is

- `fixed`: given as the size of the data,
- `tuned`: a HyperParameter to optimize, created with another value, which
  a tuner sets to the size of the data before the graph is connected,
- `tuned_on_a_copy`: the same on a deep copy of the model and its
  HyperParameter, as a tuner copies the trainer per trial; the original
  keeps its other value,
- `inferred`: left open (None) and inferred from the connected data.

The model is fed by one graph (`one`), by two graphs with data of the same
size (`two_agreeing`) or of different sizes (`two_disagreeing`). Where the
places agree, every graph sets up and runs, and the size is that of the
data; where they disagree, connecting the second graph fails. Cases that
do not hold yet are listed in GAPS with their reason.
"""

from copy import deepcopy
from typing import Any, Callable

import pytest
import torch

from qewton.algorithms.dl_models.convolutions.encoding import ConvolutionalEncoder
from qewton.algorithms.dl_models.deeponet.deeponet_fcn import FCNDeepONet
from qewton.algorithms.dl_models.fcn import FCN
from qewton.algorithms.dl_models.fno import FNO
from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.errors import DataConfigMismatchError
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node, OutputPort
from qewton.optim.parameters.helpers import HyperParameterState
from qewton.optim.parameters.number_hyperparameter import DiscreteHyperparameter

BATCH = 4


class TensorSource(Node):
    """A node providing a fixed tensor, declared with batch, feature and
    geometry axes of its shape."""

    def __init__(self, name: str, value: torch.Tensor, kinds: str) -> None:
        super().__init__(name=name)
        self.value = value
        axes = {"B": BatchAxes, "F": FeatureAxes, "G": GeometryAxes}
        config = DataConfiguration(
            *(axes[kind](size) if kind == "B" else axes[kind](shape=(size,))
              for kind, size in zip(kinds, value.shape)),
            dtype=torch.Tensor,
        )
        self._output_ports = [OutputPort(config, node=self, name="out")]

    def forward(self) -> Any:
        return self.value


def _fcn(size):
    return FCN(in_neurons=size, hidden_neurons=4, out_neurons=2, n_hidden_layers=1)


def _fcn_data(n):
    return [("x", torch.rand(BATCH, n), "BF")]


def _deeponet(size):
    # The branch size is never given: it is always inferred.
    return FCNDeepONet(
        trunk_input=1, branch_input=1, output=1, trunk_hidden_neurons=4,
        branch_hidden_neurons=4, trunk_hidden_layers=1, branch_hidden_layers=1,
        intermediate_neurons=3,
    )


def _deeponet_data(n):
    # n sensors of the branch, other points of the trunk
    return [
        ("branch", torch.rand(BATCH, n, 1), "BGF"),
        ("trunk", torch.rand(BATCH, 5, 1), "BGF"),
    ]


def _encoding(size):
    return ConvolutionalEncoder(
        in_channels=1, channels=(2,), out_channels=3, conv_kernel_size=(3,),
        input_shape=None if size is None else (size,),
    )


def _encoding_data(n):
    return [("x", torch.rand(BATCH, 1, n), "BFG")]


def _fno(size):
    return FNO(n_modes=(4,), in_channels=size, out_channels=1, hidden_channels=4, n_layers=1)


def _fno_data(n):
    return [("x", torch.rand(BATCH, n, 16), "BFG")]


def _value(size) -> Any:
    return getattr(size, "value", size)


#: model -> (factory taking the size, data of a size, the size of a model,
#: the size of the data, origins that apply)
TUNABLE = ("fixed", "tuned", "tuned_on_a_copy", "inferred")

SIZES: dict[str, tuple[Callable, Callable, Callable, int, tuple[str, ...]]] = {
    "fcn": (_fcn, _fcn_data, lambda m: _value(m.in_neurons), 3, TUNABLE),
    "deeponet": (_deeponet, _deeponet_data, lambda m: _value(m.branch_size), 6, ("inferred",)),
    "encoding": (_encoding, _encoding_data, lambda m: _value(m.input_shape[0]), 12, TUNABLE),
    "fno": (_fno, _fno_data, lambda m: _value(m.in_channels), 2, TUNABLE),
}

PLACES = ("one", "two_agreeing", "two_disagreeing")


def _model(name: str, origin: str) -> Node:
    """The model, its size given as `origin` says; a tuned size is set by
    the tuner (the HyperParameter changes, the model is built again)."""
    factory, _, _, n, _ = SIZES[name]
    if origin == "fixed":
        return factory(n)
    if origin == "inferred":
        return factory(None)
    parameter = DiscreteHyperparameter(
        (1, 64), n + 1, state=HyperParameterState.OPTIMIZE, name=f"{name} input size"
    )
    model = factory(parameter)
    if origin == "tuned_on_a_copy":
        model, parameter = deepcopy((model, parameter))
    parameter.set_value(n)
    model.reset()
    return model


def _graph(model: Node, name: str, n: int) -> Graph:
    """A graph feeding `model` with data of size `n`, set up."""
    _, data, _, _, _ = SIZES[name]
    graph = Graph()
    for (label, value, kinds), port in zip(data(n), model.input_ports):
        graph.connect(TensorSource(label, value, kinds), port)
    graph.setup()
    return graph


def _cases():
    for name, (_, _, _, _, origins) in SIZES.items():
        for origin in origins:
            for places in PLACES:
                key = (name, origin, places)
                marks = (
                    [pytest.mark.xfail(strict=True, reason=GAPS[key])] if key in GAPS else []
                )
                yield pytest.param(name, origin, places, id="-".join(key), marks=marks)


GAPS: dict[tuple[str, str, str], str] = {}


@pytest.mark.parametrize("name, origin, places", _cases())
def test_sizes(name, origin, places):
    _, _, size_of, n, _ = SIZES[name]
    model = _model(name, origin)
    first = _graph(model, name, n)
    if places == "two_disagreeing":
        with pytest.raises(DataConfigMismatchError):
            _graph(model, name, n + 2)
        return
    graphs = [first] if places == "one" else [first, _graph(model, name, n)]
    for graph in graphs:
        graph.run()
        assert graph.value_of(model.output_ports[0]) is not None
    assert size_of(model) == n


def test_a_tuner_changes_a_size_of_a_connected_model():
    """As in tuning: the graph is connected once, every trial sets a
    HyperParameter, which makes the model unbuilt, and sets up again."""
    hidden = DiscreteHyperparameter(
        (1, 64), 4, state=HyperParameterState.OPTIMIZE, name="hidden neurons"
    )
    model = FCN(in_neurons=3, hidden_neurons=hidden, out_neurons=2, n_hidden_layers=1)
    graph = _graph(model, "fcn", 3)
    for value in (6, 5):
        hidden.set_value(value)
        model.reset()
        graph.setup()
        graph.run()
        assert graph.value_of(model.output_ports[0]).shape == (BATCH, 2)
        shapes = {tuple(group.parameters.shape) for group in model.trainable_parameters}
        assert (3, value) in shapes


@pytest.mark.parametrize("name", SIZES)
def test_an_open_size_is_required_before_building(name):
    """A model whose size is still open can not be built: neither on its
    own nor in a graph that does not feed it."""
    factory, _, _, _, _ = SIZES[name]
    model = factory(None)
    with pytest.raises(RuntimeError, match="can not be built"):
        model.setup()
    graph = Graph()
    graph.add_node(model)
    with pytest.raises(RuntimeError, match="can not be built"):
        graph.setup()
