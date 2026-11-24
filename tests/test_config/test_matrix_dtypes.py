"""Test matrix of dtypes in matching: provided dtype × required dtype.

A source provides data of one dtype, a sink requires another (config T1).
Matching keeps the more special of two compatible dtypes, from general to
special: `Any`, `TensorType` (resolved to the tensor type of the node's
backend), the backend's tensor type; incompatible ones are an error. The
static dtypes of both ports stay as declared. The order itself, with
unions (e.g. `TensorType | float`) and element dtypes (`Float32`, …), is
checked on the matching core (`merge_dtypes`, `merge_element_dtypes`).
"""

from typing import Any

import pytest
import torch
from declared_nodes import B, F, sink, source

from qewton.backends import TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.config.dtypes import Complex64, Float32, Number, merge_dtypes, merge_element_dtypes
from qewton.config.errors import DataConfigMismatchError
from qewton.graphs.graphs import Graph

ERROR = "error"

#: (provided, required) -> matched dtype or ERROR
PAIRS = {
    ("any", "any"): (Any, Any, Any),
    ("any", "tensor_type"): (Any, TensorType, torch.Tensor),
    ("tensor_type", "any"): (TensorType, Any, torch.Tensor),
    ("tensor_type", "torch"): (TensorType, torch.Tensor, torch.Tensor),
    ("torch", "torch"): (torch.Tensor, torch.Tensor, torch.Tensor),
    ("float", "any"): (float, Any, float),
    ("torch", "float"): (torch.Tensor, float, ERROR),
    ("tensor_type", "float"): (TensorType, float, ERROR),
}

GAPS: dict = {}


def _config(dtype):
    return DataConfiguration(B(4), F(3), dtype=dtype)


def _cases():
    for pair in PAIRS:
        reason = GAPS.get(pair)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        yield pytest.param(pair, marks=marks, id="-".join(pair))


@pytest.mark.parametrize("pair", _cases())
def test_dtypes(pair):
    provided, required, matched = PAIRS[pair]
    data, consumer = source("data", _config(provided)), sink("loss", _config(required))
    graph = Graph()
    if matched == ERROR:
        with pytest.raises(DataConfigMismatchError):
            graph.connect(data.output_ports[0], consumer.input_ports[0])
        return
    graph.connect(data.output_ports[0], consumer.input_ports[0])
    assert consumer.input_ports[0].get_data_configuration(graph).dtype is matched
    assert data.output_ports[0].get_data_configuration(graph).dtype is matched
    assert data.output_ports[0].data_configuration.dtype is provided
    assert consumer.input_ports[0].data_configuration.dtype is required


#: (first, second) -> merged dtype or ERROR, for the matching core
DTYPE_ORDER = {
    "union_with_tensor": (TensorType | float, torch.Tensor, torch.Tensor),
    "union_with_scalar": (TensorType | float, float, float),
    "union_with_union": (TensorType | float, TensorType | int, TensorType | int),
    "int_where_float_is_expected": (TensorType | float, int, int),
    "float_where_int_is_expected": (int, float, int),
    "union_without_match": (TensorType | float, str, ERROR),
    "none_is_any": (None, float, float),
}

#: (first, second) -> merged element dtype or ERROR
ELEMENT_ORDER = {
    "open": (None, Float32, Float32),
    "number": (Number, Complex64, Complex64),
    "same": (Float32, Float32, Float32),
    "different": (Float32, Complex64, ERROR),
    "both_open": (None, None, None),
}


@pytest.mark.parametrize("name", DTYPE_ORDER)
def test_dtype_order(name):
    first, second, merged = DTYPE_ORDER[name]
    _check(merge_dtypes, first, second, merged)


@pytest.mark.parametrize("name", ELEMENT_ORDER)
def test_element_dtype_order(name):
    first, second, merged = ELEMENT_ORDER[name]
    _check(merge_element_dtypes, first, second, merged)


def _check(merge, first, second, merged):
    if merged == ERROR:
        with pytest.raises(DataConfigMismatchError):
            merge(first, second)
        return
    assert merge(first, second) == merged
    assert merge(second, first) == merged
