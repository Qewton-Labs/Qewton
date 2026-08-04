from __future__ import annotations

import json
import inspect
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

from qewton.graphs.nodes import Node, NodeConfig
from qewton.graphs.graphs import Graph
from qewton.backends.base import ComputingBackend
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase
from qewton.backends import BACKEND_DICT, Backend
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.saving.hyperparameter_codec import (
    encode_value,
    serialize_hyperparameter,
    _add_hp_to_collection,
)
from qewton.graphs.saving.schema import (
    DIR_CONSTANTS,
    DIR_NESTED_GRAPHS,
    DIR_NODES,
    DIR_TRAINABLE_PARAMETERS,
    FILE_CONFIG,
    FILE_HYPERPARAMETERS,
    KEY_BACKEND_KEY,
    KEY_CLASS,
    KEY_EDGES,
    KEY_FROM_NODE_ID,
    KEY_FROM_OUTSIDE,
    KEY_FROM_PORT,
    KEY_HP_KEY,
    KEY_HYPERPARAMETERS,
    KEY_MODE,
    KEY_NESTED_GRAPHS,
    KEY_NODE_ID,
    KEY_NODE_IDENTIFIER,
    KEY_NODES_INCLUDED,
    KEY_OBJECT_TYPE,
    KEY_OTHER_ARGS,
    KEY_PATH,
    KEY_SCHEMA_VERSION,
    KEY_SORTED,
    KEY_STATE,
    KEY_TO_NODE_ID,
    KEY_TO_OUTSIDE,
    KEY_TO_PORT,
    KEY_TYPE,
    KEY_VALUES,
    OBJECT_TYPE_GRAPH,
    OBJECT_TYPE_NODE,
    SCHEMA_VERSION,
    TYPE_BACKEND_REF,
    TYPE_CONSTANT_REF,
    TYPE_HP_REF,
    TYPE_SET,
    TYPE_TRAINABLE_PARAMETER_REF,
    TYPE_TUPLE,
)

logger = logging.getLogger(__name__)


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return {KEY_TYPE: TYPE_TUPLE, KEY_VALUES: [_jsonify(v) for v in value]}
    if isinstance(value, set):
        return {KEY_TYPE: TYPE_SET, KEY_VALUES: [_jsonify(v) for v in value]}
    if isinstance(value, type):
        return {
            KEY_CLASS: value.__name__,
            "module": value.__module__,
        }
    return repr(value)


def _save_trainable_parameters(
    value: _TrainableParameterBase,
    parameters_dir: Path,
    root_dir: Path,
    file_counter: list[int],
    backend: ComputingBackend,
) -> dict[str, Any] | list[dict[str, Any]]:
    # Save each parameter group separately to keep files small and composable.
    rel_paths: list[dict[str, Any]] = []
    for group in value:
        file_name = f"param_{file_counter[0]}_node_{group.node_id}"
        file_counter[0] += 1
        param_path = parameters_dir / file_name
        backend.save(group.parameters, param_path)
        rel_paths.append(
            {
                KEY_TYPE: TYPE_TRAINABLE_PARAMETER_REF,
                KEY_PATH: str(param_path.relative_to(root_dir)),
            }
        )

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
        return {
            KEY_TYPE: TYPE_CONSTANT_REF,
            KEY_PATH: str(param_path.relative_to(root_dir)),
        }
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
    if isinstance(value, list):
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
    if isinstance(value, (tuple, set)):
        return {
            KEY_TYPE: type(value).__name__,
            KEY_VALUES: [
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
            ],
        }
    if inspect.isclass(value) and issubclass(value, Backend):
        return {
            KEY_TYPE: TYPE_BACKEND_REF,
            KEY_BACKEND_KEY: next(k for k, v in BACKEND_DICT.items() if v == value),
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


#############################################################################


def save(obj: Node | Graph, path: str | Path, replace: bool = False) -> None:
    """Saves a Node or Graph to a file.

    Args:
        obj (Node | Graph): The Node or Graph to save.
        path (str): The path to the file where the object will be saved.
    """
    if Path(path).exists():
        if replace:
            # delete the existing directory and its contents
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"The path {path} already exists. Use replace=True \
                    to allow to overwrite it.")
    logger.info("Saving %s to %s", obj.__class__.__name__, path)
    if isinstance(obj, Node):
        save_node(obj.config_dict(), path, obj.backend)
    elif isinstance(obj, Graph):
        save_graph(obj, path)
    else:
        raise TypeError(f"Object of type {type(obj)} is not supported for saving.")
    logger.info("Saving completed")


def save_node(
    node_config: NodeConfig,
    path: str | Path,
    backend: type[Backend],
    hp_collection: dict[str, HyperParameter] | None = None,
):
    """Saves a Node to a file.

    Args:
        node (Node): The Node to save.
        path (str): The path to the file where the Node will be saved.
        hp_id_to_key (dict[str, HyperParameter] ): Optional mapping from HP id()
            to shared-file key. When provided the node's hyperparameters are
            stored as references into the graph-level hyperparameters.json
            instead of a local file.
    """
    given_collection = True
    if hp_collection is None:
        given_collection = False
        hp_collection = {}

    # Build path and create directories if they don't exist
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir = output_dir / DIR_TRAINABLE_PARAMETERS
    parameters_dir.mkdir(parents=True, exist_ok=True)

    constants_dir = output_dir / DIR_CONSTANTS
    constants_dir.mkdir(parents=True, exist_ok=True)
    # Serialize the node configuration and save it to a JSON file
    file_counter = [0]
    file_constants_counter = [0]

    config_payload = {
        KEY_SCHEMA_VERSION: SCHEMA_VERSION,
        KEY_OBJECT_TYPE: OBJECT_TYPE_NODE,
        KEY_NODE_IDENTIFIER: node_config.node_identifier,
        KEY_NODE_ID: node_config.node_id,
        KEY_MODE: node_config.mode.name,
        KEY_STATE: node_config.state.name,
    }
    # Serialize arguments, including trainable parameters,
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
    config_payload[KEY_OTHER_ARGS] = serialized_other_args

    if file_counter[0] == 0:
        # Remove the parameters directory if no parameters were saved
        parameters_dir.rmdir()

    if file_constants_counter[0] == 0:
        # Remove the parameters directory if no parameters were saved
        constants_dir.rmdir()

    # check for nested graphs inside the node:
    if len(node_config.nested_graphs) > 0:
        graphs_dir = output_dir / DIR_NESTED_GRAPHS
        graphs_dir.mkdir(parents=True, exist_ok=True)
        config_payload[KEY_NESTED_GRAPHS] = {}
        for graph_name, graph in node_config.nested_graphs.items():
            graph_path = graphs_dir / graph_name
            save_graph(graph, graph_path, hp_collection=hp_collection)
            config_payload[KEY_NESTED_GRAPHS][graph_name] = str(
                graph_path.relative_to(output_dir)
            )

    # Hyperparameters are saved either as inline references into the graph-level
    # shared file (when hp_collection is given) or in the node's own local file.
    if len(node_config.hyperparameters) > 0:
        # Add all hyperparameters to the collection
        for hp in node_config.hyperparameters.values():
            if isinstance(hp, (list, tuple)):
                for sub_hp in hp:
                    _add_hp_to_collection(sub_hp, hp_collection, node_config.node_id)
            else:
                _add_hp_to_collection(hp, hp_collection, node_config.node_id)
        # Build the reference:
        config_payload[KEY_HYPERPARAMETERS] = {}
        for name, hp in node_config.hyperparameters.items():
            if isinstance(hp, list):
                config_payload[KEY_HYPERPARAMETERS][name] = [sub_hp.name for sub_hp in hp]
            elif isinstance(hp, tuple):
                config_payload[KEY_HYPERPARAMETERS][name] = {
                    KEY_TYPE: TYPE_TUPLE,
                    KEY_VALUES: [sub_hp.name for sub_hp in hp],
                }
            else:
                config_payload[KEY_HYPERPARAMETERS][name] = hp.name
        # Save if not handled on the level above
        if not given_collection:
            hyperparameters_payload = {
                hp.name: _serialize_hyperparameter_entry(hp)
                for hp in hp_collection.values()
            }
            with (output_dir / FILE_HYPERPARAMETERS).open("w", encoding="utf-8") as f:
                json.dump(hyperparameters_payload, f, indent=2)

    with (output_dir / FILE_CONFIG).open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)


def save_graph(
    graph: Graph, path: str | Path, hp_collection: dict[str, HyperParameter] | None = None
) -> None:
    """Saves a Graph to a file.

    Args:
        graph (Graph): The Graph to save.
        path (str): The path to the file where the Graph will be saved.
    """
    collection_was_given = True
    if hp_collection is None:
        collection_was_given = False
        hp_collection = {}

    # Build path and create directories if they don't exist
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Serialize the graph configuration and save it to a JSON file
    graph_config = graph.graph_config()

    # Improve edge readability by converting node objects to their IDs
    edges_config = []
    for e in graph_config.edges:
        edges_config.append(_edge_format_save(e))

    graph_config_payload = {
        KEY_SCHEMA_VERSION: SCHEMA_VERSION,
        KEY_OBJECT_TYPE: OBJECT_TYPE_GRAPH,
        KEY_NODES_INCLUDED: list(graph_config.node_configs.keys()),
        KEY_EDGES: edges_config,
        KEY_SORTED: graph_config.graph_was_sorted,
        KEY_FROM_OUTSIDE: [_edge_format_save(e) for e in graph_config.edges_from_outside],
        KEY_TO_OUTSIDE: [_edge_format_save(e) for e in graph_config.edges_to_outside],
    }

    node_configs = {}
    for node in graph.nodes:
        node_configs[node] = node.config_dict()

    if len(graph.nodes) > 0:
        node_dir = Path(output_dir / DIR_NODES)
        node_dir.mkdir(parents=True, exist_ok=True)

        for node in graph.nodes:
            node_path = Path(node_dir / f"node_{node.node_id}")
            save_node(
                node_config=node_configs[node],
                path=node_path,
                backend=node.backend,
                hp_collection=hp_collection,
            )

    with (output_dir / FILE_CONFIG).open("w", encoding="utf-8") as f:
        json.dump(graph_config_payload, f, indent=2)

    # Also handle the saving of the hyperparameters.json file if this is the
    # root call
    if not collection_was_given and len(hp_collection) > 0:
        hp_payload = {
            hp.name: serialize_hyperparameter(hp) for hp in hp_collection.values()
        }
        with (output_dir / FILE_HYPERPARAMETERS).open("w", encoding="utf-8") as f:
            json.dump(hp_payload, f, indent=2)


def _edge_format_save(edge: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        KEY_FROM_NODE_ID: edge[0],
        KEY_FROM_PORT: edge[1],
        KEY_TO_NODE_ID: edge[2],
        KEY_TO_PORT: edge[3],
    }
