from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from qewton.graphs.nodes import Node
from qewton.graphs.graphs import Graph
from qewton.backends.base import ComputingBackend
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase


# TODO: Move this to Hyperparameters in general...
def _serialize_hyperparameter(param: HyperParameter) -> dict[str, Any]:
    return {
        "class": param.__class__.__name__,
        "module": param.__class__.__module__,
        "name": param.name,
        "state": param.state.name,
        "parameter_range": _jsonify(param.parameter_range),
        "current_value": _jsonify(param.current_value),
        "default_grid": _jsonify(param.default_grid),
        "condition": None if param.condition is None else repr(param.condition),
    }


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, type):
        return {
            "class": value.__name__,
            "module": value.__module__,
        }
    return repr(value)


def _save_trainable_parameters(
    value: _TrainableParameterBase,
    parameters_dir: Path,
    root_dir: Path,
    file_counter: list[int],
    backend: ComputingBackend,
) -> str | list[str]:
    # Save each parameter group separately to keep files small and composable.
    rel_paths: list[str] = []
    for group in value:
        file_name = f"param_{file_counter[0]}_node_{group.node_id}"
        file_counter[0] += 1
        param_path = parameters_dir / file_name
        backend.save(group.parameters, param_path)
        rel_paths.append(str(param_path.relative_to(root_dir)))

    if len(rel_paths) == 1:
        return rel_paths[0]
    return rel_paths


def _serialize_other_args(
    value: Any,
    parameters_dir: Path,
    root_dir: Path,
    file_counter: list[int],
    backend,
) -> Any:
    if isinstance(value, _TrainableParameterBase):
        return _save_trainable_parameters(
            value, parameters_dir, root_dir, file_counter, backend
        )
    if isinstance(value, dict):
        return {
            str(k): _serialize_other_args(
                v, parameters_dir, root_dir, file_counter, backend=backend
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _serialize_other_args(
                v, parameters_dir, root_dir, file_counter, backend=backend
            )
            for v in value
        ]
    return _jsonify(value)


def save(obj: Node | Graph, path: str):
    """Saves a Node or Graph to a file.

    Args:
        obj (Node | Graph): The Node or Graph to save.
        path (str): The path to the file where the object will be saved.
    """
    if isinstance(obj, Node):
        save_node(obj, path)


def save_node(node: Node, path: str):
    """Saves a Node to a file.

    Args:
        node (Node): The Node to save.
        path (str): The path to the file where the Node will be saved.
    """
    # Build path and create directories if they don't exist
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir = output_dir / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)

    # Serialize the node configuration and save it to a JSON file
    node_config = node.config_dict()
    file_counter = [0]

    config_payload = {
        "node_identifier": node_config.node_identifier,
        "node_id": node_config.node_id,
        "mode": node_config.mode.name,
        "state": node_config.state.name,
    }

    # Hyperparameters are saved in a separate JSON file to keep the
    # main config clean and manageable.
    if len(node_config.hyperparameters) > 0:
        config_payload["hyperparameters_file"] = "hyperparameters.json"
        hyperparameters_payload = {
            name: _serialize_hyperparameter(hp)
            for name, hp in node_config.hyperparameters.items()
        }
        with (output_dir / "hyperparameters.json").open("w", encoding="utf-8") as f:
            json.dump(hyperparameters_payload, f, indent=2)

    # Serialize other arguments, including trainable parameters,
    # and save them to the main config
    serialized_other_args = _serialize_other_args(
        node_config.other_args,
        parameters_dir,
        output_dir,
        file_counter,
        backend=node.backend,
    )
    config_payload["other_args"] = serialized_other_args

    if file_counter[0] == 0:
        # Remove the parameters directory if no parameters were saved
        parameters_dir.rmdir()

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
