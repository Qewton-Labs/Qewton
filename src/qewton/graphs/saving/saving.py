from __future__ import annotations

import json
import inspect
from enum import Enum
from pathlib import Path
from typing import Any

from qewton.graphs.nodes import Node, NodeConfig
from qewton.graphs.graphs import Graph
from qewton.backends.base import ComputingBackend
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase
from qewton.backends import BACKEND_DICT, Backend


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
    constants_dir: Path,
    file_counter: list[int],
    constants_file_counter: list[int],
    backend,
) -> Any:
    if isinstance(value, _TrainableParameterBase):
        return _save_trainable_parameters(
            value, parameters_dir, root_dir, file_counter, backend
        )
    if isinstance(value, backend.default_dtype):
        file_name = f"value_{constants_file_counter[0]}"
        constants_file_counter[0] += 1
        param_path = constants_dir / file_name
        backend.save(value, param_path)
        return str(param_path.relative_to(root_dir))
    if isinstance(value, dict):
        return {
            str(k): _serialize_other_args(
                v,
                parameters_dir,
                root_dir,
                constants_dir,
                file_counter,
                constants_file_counter,
                backend=backend,
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _serialize_other_args(
                v,
                parameters_dir,
                root_dir,
                constants_dir,
                file_counter,
                constants_file_counter,
                backend=backend,
            )
            for v in value
        ]
    if inspect.isclass(value) and issubclass(value, Backend):
        return next(k for k, v in BACKEND_DICT.items() if v == value)
    return _jsonify(value)


#############################################################################


def save(obj: Node | Graph, path: str | Path) -> None:
    """Saves a Node or Graph to a file.

    Args:
        obj (Node | Graph): The Node or Graph to save.
        path (str): The path to the file where the object will be saved.
    """
    if isinstance(obj, Node):
        save_node(obj.config_dict(), path, obj.backend)
    elif isinstance(obj, Graph):
        save_graph(obj, path)
    else:
        raise TypeError(f"Object of type {type(obj)} is not supported for saving.")


def save_node(node_config: NodeConfig, path: str | Path, backend: type[Backend]):
    """Saves a Node to a file.

    Args:
        node (Node): The Node to save.
        path (str): The path to the file where the Node will be saved.
    """
    # Build path and create directories if they don't exist
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir = output_dir / "trainable_parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)

    constants_dir = output_dir / "constants"
    constants_dir.mkdir(parents=True, exist_ok=True)
    # Serialize the node configuration and save it to a JSON file
    file_counter = [0]
    file_constants_counter = [0]

    config_payload = {
        "object_type": "Node",
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
        constants_dir,
        file_counter,
        file_constants_counter,
        backend=backend,
    )
    config_payload["other_args"] = serialized_other_args

    if file_counter[0] == 0:
        # Remove the parameters directory if no parameters were saved
        parameters_dir.rmdir()

    if file_constants_counter[0] == 0:
        # Remove the parameters directory if no parameters were saved
        constants_dir.rmdir()

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)


def save_graph(graph: Graph, path: str | Path) -> None:
    """Saves a Graph to a file.

    Args:
        graph (Graph): The Graph to save.
        path (str): The path to the file where the Graph will be saved.
    """
    # Build path and create directories if they don't exist
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Serialize the graph configuration and save it to a JSON file
    graph_config = graph.graph_config()

    # Improve edge readability by converting node objects to their IDs
    edges_config = []
    for e in graph_config.edges:
        edges_config.append(
            {
                "from_node_id": e[0],
                "from_port": e[1],
                "to_node_id": e[2],
                "to_port": e[3],
            }
        )
    graph_config_payload = {
        "object_type": "Graph",
        "nodes_included": list(graph_config.node_configs.keys()),
        "edges": edges_config,
        "sorted": graph_config.graph_was_sorted,
    }

    if len(graph.nodes) > 0:
        node_dir = Path(output_dir / "nodes")
        node_dir.mkdir(parents=True, exist_ok=True)

        for node in graph.nodes:
            node_path = Path(node_dir / f"node_{node.node_id}")
            save_node(
                node_config=node.config_dict(), path=node_path, backend=node.backend
            )

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(graph_config_payload, f, indent=2)
