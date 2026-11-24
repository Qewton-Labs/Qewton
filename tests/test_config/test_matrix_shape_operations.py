"""Test matrix of the shape operations of configurations, without graphs
(config N1, N2): operation × direction.

Forward, the input is known and the shape of the result is checked.
Backward, the input is unknown (an ellipsis): the operation refines it by
new unknowns, the result is merged with a known configuration, and the
shape of the input is checked; what the result learns, the shared objects
of the input know. `None` is a size that stays unknown, `...` structure
that stays unknown. Axes that an operation keeps stay the same objects.
"""

from typing import Callable

import pytest
from declared_nodes import B, F, G, dc, unknown

from qewton.config.axes import AxesDim, FeatureAxes, GeometryAxes

#: name -> (input, operation, expected shape of the result)
FORWARD: dict[str, tuple[Callable, Callable, tuple]] = {
    "drop": (lambda: dc(B(4), G(5), F(3)), lambda x: x.drop(1), (4, 3)),
    "drop_last": (lambda: dc(B(4), G(5), F(3)), lambda x: x.drop(-1), (4, 5)),
    "squeeze": (lambda: dc(B(4), G(1), F(3)), lambda x: x.squeeze(1), (4, 3)),
    "set_size": (lambda: dc(B(4), G(5), F(3)), lambda x: x.set_size(1, 1), (4, 1, 3)),
    "unsqueeze": (lambda: dc(B(4), F(3)), lambda x: x.unsqueeze(1), (4, 1, 3)),
    "unsqueeze_last": (lambda: dc(B(4), F(3)), lambda x: x.unsqueeze(-1), (4, 3, 1)),
    "permute": (lambda: dc(B(4), F(3)), lambda x: x.permute([1, 0]), (3, 4)),
    "flatten": (lambda: dc(B(4), G(5), F(3)), lambda x: x.flatten(1), (4, 15)),
    "unflatten": (lambda: dc(B(4), F(15)), lambda x: x.unflatten(1, (5, -1)), (4, 5, 3)),
    "reshape": (lambda: dc(B(4), G(5), F(3)), lambda x: x.reshape((4, -1)), (4, 15)),
    "concat": (lambda: dc(B(4), F(3)), lambda x: x.concat([dc(B(4), F(2))], -1), (4, 5)),
    "broadcast": (lambda: dc(B(4), F(3)), lambda x: x.broadcast(dc(B(1), F(3))), (4, 3)),
    "window": (lambda: dc(B(4), F(3), G(8)), lambda x: x.window(-1, 2, stride=2), (4, 3, 4)),
}

#: name -> (operation on an unknown input, known result, expected input shape)
BACKWARD: dict[str, tuple[Callable, Callable, tuple]] = {
    "drop": (lambda x: x.drop(1), lambda: dc(B(4), F(3)), (4, None, 3)),
    "squeeze": (lambda x: x.squeeze(1), lambda: dc(B(4), F(3)), (4, 1, 3)),
    "unsqueeze": (lambda x: x.unsqueeze(1), lambda: dc(B(4), G(1), F(3)), (4, 3)),
    "permute": (lambda x: x.permute([1, 0]), lambda: dc(F(3), B(4)), (4, 3)),
    "set_size": (lambda x: x.set_size(1, 1), lambda: dc(B(4), G(1), F(3)), (4, None, 3)),
    "flatten": (lambda x: x.flatten(1), lambda: dc(B(4), F(15)), (4, ...)),
    "window_stride_1": (
        lambda x: x.window(-1, 3, stride=1, padding=1), lambda: dc(B(4), F(3), G(8)), (4, 3, 8)
    ),
    "window_stride_2": (lambda x: x.window(-1, 2, stride=2), lambda: dc(B(4), F(3), G(4)), (4, 3, None)),
}


@pytest.mark.parametrize("name", FORWARD)
def test_forward(name):
    build, operation, expected = FORWARD[name]
    result = operation(build())
    assert result.resolve().shape == expected


@pytest.mark.parametrize("name", BACKWARD)
def test_backward(name):
    operation, known, expected = BACKWARD[name]
    x = unknown()
    result = operation(x)
    result.merge(known())
    assert x.resolve().shape == expected


def test_kept_axes_stay_the_same_objects():
    x = dc(B(4), G(5), F(3))
    batch, features = x.axes[0], x.axes[2]
    result = x.drop(1).resolve()
    assert result.axes[0] is batch and result.axes[1] is features


def test_broadcasting_keeps_the_axes_of_the_larger_input():
    x, y = dc(B(4), F(3)), dc(B(1), F(3))
    result = x.broadcast(y).resolve()
    assert result.axes[0] is x.axes[0] and result.axes[0] is not y.axes[0]


def test_broadcasting_keeps_the_dims_of_one_axes_together():
    def grid():
        return GeometryAxes(shape=(AxesDim(None), AxesDim(None)))

    result = dc(B(4), grid()).broadcast(dc(B(4), grid())).resolve()
    assert len(result.axes) == 2 and isinstance(result.axes[1], GeometryAxes)


def test_broadcasting_keeps_an_axes_that_spans_dims_of_several_axes():
    # [B, 1, F(5)] against [B, F(4, 5)]: the feature axes of the second holds
    # both dims, which come from two axes of the first.
    x = dc(B(6), G(1), F(5))
    y = dc(B(6), FeatureAxes(shape=(AxesDim(4), AxesDim(5))))
    result = x.broadcast(y).resolve()
    assert result.shape == (6, 4, 5) and result.axes[1] is y.axes[1]


def test_concat_solves_a_missing_part():
    x, y = dc(B(4), F(3)), dc(B(None), F(None))
    x.concat([y], -1).merge(dc(B(4), F(5)))
    assert y.resolve().shape == (4, 2)
