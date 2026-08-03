"""Codec for serializing and deserializing HyperParameter objects."""

from __future__ import annotations

import importlib
import inspect
from enum import Enum
from pathlib import Path
from typing import Any

from qewton.optim.parameters.categorical_hyperparameter import (
    BooleanHyperparameter,
    CategoricalHyperparameter,
)
from qewton.optim.parameters.number_hyperparameter import (
    ContinuousHyperparameter,
    DiscreteHyperparameter,
)
from qewton.optim.parameters.helpers import HyperParameterState
from qewton.optim.parameters.hyperparameter_base import HyperParameter

from .schema import (
    KEY_CLASS,
    KEY_CONDITION,
    KEY_CURRENT_VALUE,
    KEY_DEFAULT_GRID,
    KEY_EXTRA_ARGS,
    KEY_MODULE,
    KEY_NAME,
    KEY_PARAMETER_RANGE,
    KEY_STATE,
    KEY_TYPE,
    KEY_VALUES,
    TYPE_CLASS_REF,
    TYPE_ENUM_REF,
    TYPE_SET,
    TYPE_TUPLE,
    KEY_BOOLEAN_HP,
    KEY_CONTINUOUS_HP,
    KEY_DISCRETE_HP,
    KEY_CATEGORICAL_HP,
)

_hp_dict = {
    BooleanHyperparameter: KEY_BOOLEAN_HP,
    CategoricalHyperparameter: KEY_CATEGORICAL_HP,
    ContinuousHyperparameter: KEY_CONTINUOUS_HP,
    DiscreteHyperparameter: KEY_DISCRETE_HP,
}
_inverse_hp_dict = {v: k for k, v in _hp_dict.items()}


def _get_class(module: str, class_name: str) -> type:
    mod = importlib.import_module(module)
    return getattr(mod, class_name)


def encode_value(value: Any) -> Any:
    """Convert Python values into JSON-safe values with typed wrappers."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {
            KEY_TYPE: TYPE_ENUM_REF,
            KEY_CLASS: value.__class__.__name__,
            KEY_MODULE: value.__class__.__module__,
            KEY_NAME: value.name,
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    if isinstance(value, tuple):
        return {KEY_TYPE: TYPE_TUPLE, KEY_VALUES: [encode_value(v) for v in value]}
    if isinstance(value, set):
        return {KEY_TYPE: TYPE_SET, KEY_VALUES: [encode_value(v) for v in value]}
    if isinstance(value, type):
        return {
            KEY_TYPE: TYPE_CLASS_REF,
            KEY_CLASS: value.__name__,
            KEY_MODULE: value.__module__,
        }
    return repr(value)


def decode_value(value: Any) -> Any:
    """Decode JSON-safe values produced by encode_value()."""
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if isinstance(value, dict):
        value_type = value.get(KEY_TYPE)
        if value_type == TYPE_TUPLE:
            return tuple(decode_value(v) for v in value[KEY_VALUES])
        if value_type == TYPE_SET:
            return set(decode_value(v) for v in value[KEY_VALUES])
        if value_type == TYPE_CLASS_REF:
            return _get_class(value[KEY_MODULE], value[KEY_CLASS])
        if value_type == TYPE_ENUM_REF:
            enum_class = _get_class(value[KEY_MODULE], value[KEY_CLASS])
            return enum_class[value[KEY_NAME]]  # type: ignore
        return {k: decode_value(v) for k, v in value.items()}
    return value


def serialize_hyperparameter(param: HyperParameter) -> dict[str, Any]:
    """Serialize a HyperParameter instance into a JSON payload."""
    payload: dict[str, Any] = {
        KEY_CLASS: _hp_dict[type(param)],
        KEY_NAME: param.name,
        KEY_STATE: param.state.name,
        KEY_PARAMETER_RANGE: encode_value(param.parameter_range),
        KEY_CURRENT_VALUE: encode_value(param.current_value),
        KEY_DEFAULT_GRID: encode_value(param.default_grid),
        KEY_CONDITION: None if param.condition is None else repr(param.condition),
    }

    base_keys = {
        "state",
        "parameter_range",
        "current_value",
        "name",
        "condition",
        "default_grid",
        "_registry",
    }
    extra_args = {
        k: encode_value(v)
        for k, v in param.__dict__.items()
        if not k.startswith("_") and k not in base_keys
    }
    if extra_args:
        payload[KEY_EXTRA_ARGS] = extra_args
    return payload


def deserialize_hyperparameter(data: dict[str, Any]) -> HyperParameter:
    """Reconstruct a HyperParameter instance from a serialized payload."""
    hp_class: type[HyperParameter] = _inverse_hp_dict[data[KEY_CLASS]]
    state = HyperParameterState[data[KEY_STATE]]
    name = data.get(KEY_NAME, "")
    current_value = decode_value(data.get(KEY_CURRENT_VALUE))
    parameter_range = decode_value(data.get(KEY_PARAMETER_RANGE, []))
    default_grid = decode_value(data.get(KEY_DEFAULT_GRID))

    sig = inspect.signature(hp_class.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}

    kwargs: dict[str, Any] = {"state": state, "name": name}

    if hp_class == BooleanHyperparameter:
        kwargs["initial_value"] = current_value
    elif "categories" in accepted:
        categories = (
            parameter_range if isinstance(parameter_range, list) else [current_value]
        )
        kwargs["categories"] = categories
        kwargs["initial_value"] = current_value
    elif "parameter_range" in accepted:
        kwargs["parameter_range"] = parameter_range
        kwargs["initial_value"] = current_value
    else:
        kwargs["parameter_range"] = parameter_range if parameter_range is not None else []
        kwargs["initial_value"] = current_value

    if "default_grid" in accepted and default_grid is not None:
        kwargs["default_grid"] = default_grid

    for key, value in data.get(KEY_EXTRA_ARGS, {}).items():
        if key in accepted and key not in kwargs:
            kwargs[key] = decode_value(value)

    hp = hp_class(**kwargs)
    hp.current_value = current_value
    return hp
