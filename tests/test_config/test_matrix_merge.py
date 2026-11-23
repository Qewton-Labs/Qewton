"""Test matrices of the matching core, without graphs (config M1–M5).

1. Every case of `ellipsis_cases`: the configurations of the ports are
   instantiated per node, each connection merges two of them, in every
   order; after resolving, the shapes are as expected, connected ports
   hold the very same axes, and a contradiction names both origins.
2. Pairs of single axes or dims: the representative after merging, or an
   error.
3. Operation dims: sizes solved backwards where unique.
4. Dims bound to HyperParameters: the value of the HyperParameter after a
   sequence of merges, or an error.
5. Ambiguous alignments: the one merging the most is preferred; several
   such wait for more to be known.
"""

from copy import deepcopy
from itertools import permutations
from typing import Callable

import pytest
from declared_nodes import B, F, G
from ellipsis_cases import CASES, port

from qewton.config import Variable
from qewton.config.axes import Axes, AxesDim, EllipsisAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.errors import DataConfigMismatchError
from qewton.optim.parameters.hyperparameter_base import HyperParameter


def _order_cases():
    for name, case in CASES.items():
        for order in permutations(range(len(case.connections))):
            yield pytest.param(name, order, id=f"{name}-{'-'.join(map(str, order))}")


@pytest.mark.parametrize("name, order", _order_cases())
def test_ellipsis_cases(name, order):
    case = CASES[name]
    nodes = case.nodes()
    configs = {}
    for node in nodes.values():
        for node_port, config in node.copy_data_configs().items():
            config.mark_origin(node.name)
            configs[node_port] = config

    def merge_all():
        for i in order:
            first, second = case.connections[i]
            configs[port(nodes, first)].merge(configs[port(nodes, second)])

    if case.error:
        with pytest.raises(DataConfigMismatchError, match="data") as error:
            merge_all()
        assert "loss" in str(error.value)
        return
    merge_all()
    for config in configs.values():
        config.resolve()
    for port_name, shape in case.expected.items():
        assert configs[port(nodes, port_name)].shape == shape
    for first, second in case.connections:
        mine, theirs = configs[port(nodes, first)].axes, configs[port(nodes, second)].axes
        assert len(mine) == len(theirs) and all(a is b for a, b in zip(mine, theirs))


X = Variable("x", 2)

#: name -> (first, second, check of the representative or "error")
PAIRS: dict[str, tuple[Callable, Callable, Callable | str]] = {
    "sizes": (lambda: AxesDim(4), lambda: AxesDim(), lambda r: r.size == 4),
    "sizes_contradict": (lambda: AxesDim(4), lambda: AxesDim(5), "error"),
    "unknown_kind_takes_kind": (lambda: Axes(None), lambda: B(4), lambda r: type(r) is type(B())),
    "kinds_contradict": (lambda: B(4), lambda: G(4), "error"),
    "feature_takes_variable": (
        lambda: FeatureAxes(shape=(2,)), lambda: FeatureAxes(X), lambda r: r.variables is X
    ),
    "variables_contradict": (
        lambda: FeatureAxes(Variable("a", 1)), lambda: FeatureAxes(Variable("b", 1)), "error"
    ),
    "variables_agree_with_their_components": (
        lambda: FeatureAxes(X * Variable("t", 1)),
        lambda: FeatureAxes(Variable.compose([Variable.compose(X.children), Variable("t", 1)])),
        lambda r: r.shape[0].size == 3,
    ),
    "variables_in_another_order_contradict": (
        lambda: FeatureAxes(X * Variable("t", 1)),
        lambda: FeatureAxes(Variable("t", 1) * X),
        "error",
    ),
    "feature_dims_merge": (lambda: F(None), lambda: F(3), lambda r: r.shape[0].size == 3),
    "feature_dims_contradict": (lambda: F(2), lambda: F(3), "error"),
}


@pytest.mark.parametrize("name", PAIRS)
def test_pairs(name):
    first, second, check = PAIRS[name]
    a, b = first(), second()
    if check == "error":
        with pytest.raises(DataConfigMismatchError):
            a.merge(b)
        return
    representative = a.merge(b)
    assert a.representative is b.representative is representative
    assert check(representative)


def _sum():
    n, m = AxesDim(), AxesDim()
    total = n + m
    return {"n": n, "m": m, "total": total}


def _product():
    n, m = AxesDim(), AxesDim()
    return {"n": n, "m": m, "total": n * m}


#: name -> (dims, merges as (dim name, size), expected sizes)
OPERATIONS = {
    "sum_backwards": (_sum, [("total", 10), ("n", 4)], {"m": 6}),
    "sum_backwards_other_order": (_sum, [("n", 4), ("total", 10)], {"m": 6}),
    "sum_forwards": (_sum, [("n", 4), ("m", 6)], {"total": 10}),
    "product_backwards": (_product, [("total", 12), ("m", 3)], {"n": 4}),
    "product_open": (_product, [("total", 12)], {"n": None, "m": None}),
}


@pytest.mark.parametrize("name", OPERATIONS)
def test_operation_dims(name):
    build, merges, expected = OPERATIONS[name]
    dims = build()
    for dim_name, size in merges:
        dims[dim_name].merge(AxesDim(size))
    assert {k: dims[k].size for k in expected} == expected


def _open():
    return HyperParameter.inferred("open size")


def _fixed():
    return HyperParameter.from_value(3, "fixed size")


def _merged_in_turn(parameter, *dims):
    """Merge the dims one after another into the first, which is bound to
    `parameter`."""
    first = AxesDim(parameter)
    for dim in dims:
        first.merge(dim)
    return parameter


def _two_places(sizes):
    """Two copies of a dim bound to one open parameter (two places of one
    node), each merged with a size."""
    parameter = _open()
    declared = AxesDim(parameter)
    for size in sizes:
        deepcopy(declared).merge(AxesDim(size))
    return parameter


def _through_an_unknown():
    parameter, unknown = _open(), AxesDim()
    AxesDim(parameter).merge(unknown)
    unknown.merge(AxesDim(5))
    return parameter


def _operation_is_read():
    parameter = _open()
    AxesDim(parameter * HyperParameter.from_value(2, "factor")).merge(AxesDim(6))
    return parameter


def _computed_later():
    """An open parameter merged into a product whose operand becomes known
    only afterwards (e.g. a flattened sequence with an unbound ellipsis)."""
    parameter, n = _open(), AxesDim()
    (n * AxesDim(2)).merge(AxesDim(parameter))
    n.merge(AxesDim(3))
    return parameter


#: name -> (merges returning the parameter, its expected value or "error")
BOUND = {
    "open_is_inferred": (lambda: _merged_in_turn(_open(), AxesDim(3)), 3),
    "open_stays_open": (lambda: _merged_in_turn(_open(), AxesDim()), None),
    "open_through_an_unknown": (_through_an_unknown, 5),
    "fixed_agrees": (lambda: _merged_in_turn(_fixed(), AxesDim(3)), 3),
    "fixed_contradicts": (lambda: _merged_in_turn(_fixed(), AxesDim(4)), "error"),
    "fixed_gives_its_size": (lambda: _merged_in_turn(_fixed(), AxesDim()), 3),
    "two_places_agree": (lambda: _two_places([3, 3]), 3),
    "two_places_contradict": (lambda: _two_places([3, 4]), "error"),
    "operation_is_read_not_solved": (_operation_is_read, None),
    "inferred_once_an_operation_is_computed": (_computed_later, 6),
}


@pytest.mark.parametrize("name", BOUND)
def test_bound_dims(name):
    merges, expected = BOUND[name]
    if expected == "error":
        with pytest.raises(DataConfigMismatchError):
            merges()
        return
    assert merges().value == expected


def test_a_fixed_size_flows_to_merged_dims():
    unknown = AxesDim()
    AxesDim(_fixed()).merge(unknown)
    assert unknown.size == 3


def test_a_changed_hyperparameter_changes_the_size_of_merged_dims():
    size = HyperParameter.from_value(4, "batch size")
    unknown, product = AxesDim(), AxesDim()
    AxesDim(size).merge(unknown)
    double = unknown * AxesDim(2)
    product.merge(double)
    size.set_value(9)
    assert unknown.size == 9 and product.size == 18


def test_a_contradiction_names_the_hyperparameter():
    with pytest.raises(DataConfigMismatchError, match="fixed size"):
        AxesDim(_fixed()).merge(AxesDim(4))


def test_a_waiting_equation_is_merged_once_an_ellipsis_is_bound():
    e1, e2 = EllipsisAxes(), EllipsisAxes()
    first = DataConfiguration(B(4), B(), e1, F(3))
    second = DataConfiguration(e2, B(), B(), F(3))
    first.merge(second)  # the batch axes may overlap by one or by two: it waits
    assert e1.waiting and e2.waiting
    DataConfiguration(e1).merge(DataConfiguration())  # e1 is empty
    first.resolve(), second.resolve()
    assert first.shape == second.shape == (4, None, 3)
    assert all(a is b for a, b in zip(first.axes, second.axes))


def test_the_alignment_merging_the_most_is_preferred():
    first = DataConfiguration(B(4), G(28), EllipsisAxes())
    second = DataConfiguration(EllipsisAxes(), G(28), F(1))
    first.merge(second)
    assert first.resolve().shape == second.resolve().shape == (4, 28, 1)
