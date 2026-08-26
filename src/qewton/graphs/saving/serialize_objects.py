from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

from qewton.graphs.nodes import InputPort, Node, OutputPort, Port
from qewton.backends.base import ComputingBackend
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase
from qewton.backends import BACKEND_DICT, Backend
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.config.axes import Axes, AxesDim
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.graphs.saving.hyperparameter_saving import (
    encode_value,
    serialize_hyperparameter,
)
from qewton.graphs.saving.serialize_dataconfigurations import (
    DataConfigSerializationResult,
)
from qewton.geometries.base import Geometry
from qewton.graphs.saving.schema import (
    DIR_PARAMETERS,
    KEY_BACKEND_KEY,
    KEY_HP_KEY,
    KEY_NAME,
    KEY_NODE_ID,
    KEY_PATH,
    KEY_TYPE,
    KEY_VALUES,
    TYPE_BACKEND_REF,
    TYPE_CONSTANT_REF,
    TYPE_HP_REF,
    TYPE_TRAINABLE_PARAMETER_REF,
    TYPE_TUPLE,
    TYPE_NODE,
    TYPE_VARIABLE,
    TYPE_GEOMETRY,
    TYPE_ELLIPSIS,
    TYPE_INPUT_PORT,
    TYPE_OUTPUT_PORT,
    KEY_PORT_DEFAULT,
    KEY_AXES,
    KEY_AXES_DIMENSIONS,
    KEY_DATA_CONFIGURATIONS,
    TYPE_DATACONFIG,
)


@dataclass
class SaveContext:
    parameters_dir: Path
    file_counter: int
    constants_file_counter: int
    node_save_dict: dict[int, Any]
    graph_save_dict: dict
    hp_collection: dict[str, HyperParameter]
    input_port_map: dict[int, int]
    output_port_map: dict[int, int]
    data_cdf_seri: DataConfigSerializationResult


def _save_trainable_parameters(
    value: _TrainableParameterBase,
    save_context: SaveContext,
    backend: ComputingBackend,
) -> dict[str, Any] | list[dict[str, Any]]:
    # Save each parameter group separately to keep files small and composable.
    rel_paths: list[dict[str, Any]] = []
    for group in value:
        file_name = f"param_{save_context.file_counter}"
        save_context.file_counter += 1
        param_path = save_context.parameters_dir / file_name
        backend.save(group.parameters, param_path)
        rel_paths.append(
            {
                KEY_TYPE: TYPE_TRAINABLE_PARAMETER_REF,
                KEY_PATH: DIR_PARAMETERS + "/" + file_name,
            }
        )

    if len(rel_paths) == 1:
        return rel_paths[0]
    return rel_paths


def _serialize_other_args(
    value: Any,
    save_context: SaveContext,
    node_dependencies: list[Node],
    backend,
) -> Any:
    if isinstance(value, _TrainableParameterBase):
        return _save_trainable_parameters(value, save_context, backend)
    if value is ...:
        return {KEY_TYPE: TYPE_ELLIPSIS}
    if isinstance(value, Geometry):
        raw = value.save()
        serialized_fields = {}
        for k, v in raw.items():
            serialized_fields[k] = _serialize_other_args(
                v,
                save_context=save_context,
                backend=backend,
                node_dependencies=node_dependencies,
            )
        return {KEY_TYPE: TYPE_GEOMETRY, KEY_VALUES: serialized_fields}
    if isinstance(value, Variable):
        return {KEY_TYPE: TYPE_VARIABLE, KEY_VALUES: dict(value)}
    if isinstance(value, backend.default_dtype):
        file_name = f"value_{save_context.constants_file_counter}"
        save_context.constants_file_counter += 1
        param_path = save_context.parameters_dir / file_name
        backend.save(value, param_path)
        return {
            KEY_TYPE: TYPE_CONSTANT_REF,
            KEY_PATH: DIR_PARAMETERS + "/" + file_name,
        }
    if isinstance(value, dict):
        return {
            str(k): _serialize_other_args(
                v,
                save_context=save_context,
                node_dependencies=node_dependencies,
                backend=backend,
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _serialize_other_args(
                v,
                save_context=save_context,
                node_dependencies=node_dependencies,
                backend=backend,
            )
            for v in value
        ]
    if isinstance(value, (tuple, set)):
        return {
            KEY_TYPE: type(value).__name__,
            KEY_VALUES: [
                _serialize_other_args(
                    v,
                    save_context=save_context,
                    node_dependencies=node_dependencies,
                    backend=backend,
                )
                for v in value
            ],
        }
    if inspect.isclass(value) and issubclass(value, Backend):
        key = next((k for k, v in BACKEND_DICT.items() if v == value), None)
        if key is None:
            raise ValueError(f"Backend class {value!r} not found in BACKEND_DICT.")
        return {KEY_TYPE: TYPE_BACKEND_REF, KEY_BACKEND_KEY: key}
    if isinstance(value, Node):
        node_dependencies.append(value)
        return {KEY_TYPE: TYPE_NODE, KEY_NODE_ID: value.node_id}
    if isinstance(value, InputPort):
        return {
            KEY_TYPE: TYPE_INPUT_PORT,
            KEY_NAME: save_context.input_port_map[id(value)],
        }
    if isinstance(value, OutputPort):
        return {
            KEY_TYPE: TYPE_OUTPUT_PORT,
            KEY_NAME: save_context.output_port_map[id(value)],
        }
    if isinstance(value, DataConfiguration):
        return {
            KEY_TYPE: KEY_DATA_CONFIGURATIONS,
            KEY_NAME: save_context.data_cdf_seri.config_list.index(value),
        }
    if isinstance(value, Axes):
        return {
            KEY_TYPE: KEY_AXES,
            KEY_NAME: save_context.data_cdf_seri.axes_list.index(value),
        }
    if isinstance(value, AxesDim):
        return {
            KEY_TYPE: KEY_AXES_DIMENSIONS,
            KEY_NAME: save_context.data_cdf_seri.axes_dim_list.index(value),
        }
    return encode_value(value)


def _serialize_hyperparameter_entry(value: Any, hp_id_to_key: dict | None = None) -> Any:
    if isinstance(value, HyperParameter):
        if hp_id_to_key is not None and id(value) in hp_id_to_key:
            return {KEY_TYPE: TYPE_HP_REF, KEY_HP_KEY: hp_id_to_key[id(value)]}
        return serialize_hyperparameter(value)
    if isinstance(value, list):
        return [_serialize_hyperparameter_entry(v, hp_id_to_key) for v in value]
    if isinstance(value, tuple):
        return {
            KEY_TYPE: TYPE_TUPLE,
            KEY_VALUES: [_serialize_hyperparameter_entry(v, hp_id_to_key) for v in value],
        }
    raise TypeError(f"Unsupported hyperparameter payload type: {type(value)}")


def serialize_port(port: Port, idx: int, save_context: SaveContext) -> dict[str, Any]:
    """Serializes a Port object into a dictionary for saving.

    Args:
        port (Port): The Port object to serialize.

    Returns:
        dict[str, Any]: A dictionary representation of the Port object.
    """
    port_data = {
        TYPE_DATACONFIG: save_context.data_cdf_seri.config_list.index(
            port.data_configuration
        ),
        KEY_NAME: port.name,
    }
    if isinstance(port, InputPort):
        save_context.input_port_map[id(port)] = idx
        if not port.is_required:
            port_data[KEY_PORT_DEFAULT] = _serialize_other_args(
                port.default,
                save_context=save_context,
                node_dependencies=[],
                backend=port.node.backend,
            )
    elif isinstance(port, OutputPort):
        save_context.output_port_map[id(port)] = idx

    return port_data
