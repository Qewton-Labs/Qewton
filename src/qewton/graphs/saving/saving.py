from __future__ import annotations

import json
import inspect
import logging
import shutil
from pathlib import Path
from typing import Any

from qewton.graphs.nodes import Node, NodeConfig
from qewton.graphs.graphs import Graph
from qewton.backends.base import ComputingBackend
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase
from qewton.backends import BACKEND_DICT, Backend
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.config.variables import Variable
from qewton.graphs.saving.hyperparameter_saving import (
    encode_value,
    serialize_hyperparameter,
    _add_hp_to_collection,
)
from qewton.geometries.base import Geometry
from qewton.graphs.saving.schema import (
    DIR_PARAMETERS,
    FILE_NODES,
    FILE_GRAPHS,
    FILE_CONFIG,
    FILE_HYPERPARAMETERS,
    KEY_BACKEND_KEY,
    KEY_EDGES,
    KEY_FROM_NODE_ID,
    KEY_FROM_OUTSIDE,
    KEY_FROM_PORT,
    KEY_HP_KEY,
    KEY_HYPERPARAMETERS,
    KEY_MODE,
    KEY_NODE_ID,
    KEY_NODE_IDENTIFIER,
    KEY_NODES_INCLUDED,
    KEY_OBJECT_TYPE,
    KEY_OTHER_ARGS,
    KEY_NESTED_GRAPHS,
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
    TYPE_TRAINABLE_PARAMETER_REF,
    TYPE_TUPLE,
    TYPE_NODE,
    TYPE_VARIABLE,
    TYPE_GEOMETRY,
    TYPE_ELLIPSIS,
)

logger = logging.getLogger(__name__)


def _save_trainable_parameters(
    value: _TrainableParameterBase,
    parameters_dir: Path,
    file_counter: list[int],
    backend: ComputingBackend,
) -> dict[str, Any] | list[dict[str, Any]]:
    # Save each parameter group separately to keep files small and composable.
    rel_paths: list[dict[str, Any]] = []
    for group in value:
        file_name = f"param_{file_counter[0]}"
        file_counter[0] += 1
        param_path = parameters_dir / file_name
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
    parameters_dir: Path,
    file_counter: list[int],
    constants_file_counter: list[int],
    node_dependencies: list[Node],
    backend,
) -> Any:
    if isinstance(value, _TrainableParameterBase):
        return _save_trainable_parameters(value, parameters_dir, file_counter, backend)
    if value is ...:
        return {KEY_TYPE: TYPE_ELLIPSIS}
    if isinstance(value, Geometry):
        raw = value.save()
        serialized_fields = {}
        for k, v in raw.items():
            serialized_fields[k] = _serialize_other_args(
                v,
                parameters_dir,
                file_counter,
                constants_file_counter,
                backend=backend,
                node_dependencies=node_dependencies,
            )
        return {KEY_TYPE: TYPE_GEOMETRY, KEY_VALUES: serialized_fields}
    if isinstance(value, Variable):
        return {KEY_TYPE: TYPE_VARIABLE, KEY_VALUES: dict(value)}
    if isinstance(value, backend.default_dtype):
        file_name = f"value_{constants_file_counter[0]}"
        constants_file_counter[0] += 1
        param_path = parameters_dir / file_name
        backend.save(value, param_path)
        return {
            KEY_TYPE: TYPE_CONSTANT_REF,
            KEY_PATH: DIR_PARAMETERS + "/" + file_name,
        }
    if isinstance(value, dict):
        return {
            str(k): _serialize_other_args(
                v,
                parameters_dir,
                file_counter,
                constants_file_counter,
                node_dependencies=node_dependencies,
                backend=backend,
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _serialize_other_args(
                v,
                parameters_dir,
                file_counter,
                constants_file_counter,
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
                    parameters_dir,
                    file_counter,
                    constants_file_counter,
                    node_dependencies=node_dependencies,
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
    if isinstance(value, Node):
        node_dependencies.append(value)
        return {KEY_TYPE: TYPE_NODE, KEY_NODE_ID: value.node_id}
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

    # Setup all the necessary directories and counters for saving parameters and constants
    node_save_dict: dict[int, Any] = {}
    graph_save_dict = {}
    hp_collection: dict[str, HyperParameter] = {}
    parameter_path = Path(path) / DIR_PARAMETERS
    file_counter = [0]  # Use a list to allow mutation within nested functions
    constant_counter = [0]  # Use a list to allow mutation within nested functions
    config_dict: dict = {KEY_SCHEMA_VERSION: SCHEMA_VERSION}
    if not parameter_path.exists():
        parameter_path.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, Node):
        config_dict[KEY_OBJECT_TYPE] = OBJECT_TYPE_NODE
        config_dict[KEY_NODE_ID] = obj.node_id
        save_node(
            obj.config_dict(),
            node_save_dict,
            graph_save_dict,
            hp_collection,
            parameter_path,
            file_counter,
            constant_counter,
            obj.backend,
        )
    elif isinstance(obj, Graph):
        config_dict[KEY_OBJECT_TYPE] = OBJECT_TYPE_GRAPH
        save_graph(
            graph=obj,
            save_key=OBJECT_TYPE_GRAPH,
            node_save_dict=node_save_dict,
            graph_save_dict=graph_save_dict,
            hp_collection=hp_collection,
            parameter_path=parameter_path,
            file_counter=file_counter,
            constant_counter=constant_counter,
        )
    else:
        raise TypeError(f"Object of type {type(obj)} is not supported for saving.")
    # Save the node and graph configurations to JSON files
    with open(Path(path) / FILE_CONFIG, "w", encoding="utf-8") as f:
        f.write(json.dumps(config_dict, indent=2))
    with open(Path(path) / FILE_NODES, "w", encoding="utf-8") as f:
        f.write(json.dumps(node_save_dict, indent=4))
    with open(Path(path) / FILE_GRAPHS, "w", encoding="utf-8") as f:
        f.write(json.dumps(graph_save_dict, indent=4))
    # Save the hyperparameter collection to a JSON file
    with open(Path(path) / FILE_HYPERPARAMETERS, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {k: serialize_hyperparameter(v) for k, v in hp_collection.items()},
                indent=4,
            )
        )
    # The parameters and constants are saved in their respective files during the
    # serialization process.

    logger.info("Saving completed")


def save_node(
    node_config: NodeConfig,
    node_save_dict: dict[int, Any],
    graph_save_dict: dict[str, Any],
    hp_collection: dict[str, HyperParameter],
    parameter_path: Path,
    file_counter: list[int],
    constant_counter: list[int],
    backend: type[Backend],
):
    if node_config.node_id in node_save_dict:
        return  # Node already saved, skip to avoid duplicates

    """Saves a Node to a file."""
    # Serialize the node configuration and save it to a JSON file
    config_payload = {
        KEY_NODE_IDENTIFIER: node_config.node_identifier,
        KEY_NODE_ID: node_config.node_id,
        KEY_MODE: node_config.mode.name,
        KEY_STATE: node_config.state.name,
    }
    # Serialize arguments, including trainable parameters,
    # and save them to the main config
    node_dependencies = []
    serialized_other_args = _serialize_other_args(
        node_config.other_args,
        parameter_path,
        file_counter,
        constant_counter,
        node_dependencies=node_dependencies,
        backend=backend,
    )
    config_payload[KEY_OTHER_ARGS] = serialized_other_args

    # check for nested graphs inside the node:
    if len(node_config.nested_graphs) > 0:
        config_payload[KEY_NESTED_GRAPHS] = {}
        for graph_name, graph in node_config.nested_graphs.items():
            save_name = f"{graph_name}_id_{node_config.node_id}"
            save_graph(
                graph=graph,
                save_key=save_name,
                node_save_dict=node_save_dict,
                graph_save_dict=graph_save_dict,
                hp_collection=hp_collection,
                parameter_path=parameter_path,
                file_counter=file_counter,
                constant_counter=constant_counter,
            )
            config_payload[KEY_NESTED_GRAPHS][graph_name] = save_name

    # save all other nodes that are dependencies of this node
    for dep_node in node_dependencies:
        save_node(
            node_config=dep_node.config_dict(),
            node_save_dict=node_save_dict,
            graph_save_dict=graph_save_dict,
            hp_collection=hp_collection,
            parameter_path=parameter_path,
            file_counter=file_counter,
            constant_counter=constant_counter,
            backend=dep_node.backend,
        )

    # Save the node configuration to the node_save_dict, ensuring no duplicate node_ids
    node_save_dict[node_config.node_id] = config_payload

    # Hyperparameters are saved on the global level, so we dont have duplicates
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


def save_graph(
    graph: Graph,
    save_key: str,
    node_save_dict: dict[int, Any],
    graph_save_dict: dict[str, Any],
    hp_collection: dict[str, HyperParameter],
    parameter_path: Path,
    file_counter: list[int],
    constant_counter: list[int],
) -> None:
    """Saves a Graph to a file.

    Args:
        graph (Graph): The Graph to save.
        path (str): The path to the file where the Graph will be saved.
    """
    # Serialize the graph configuration and save it to a JSON file
    graph_config = graph.graph_config()

    # Improve edge readability by converting node objects to their IDs
    edges_config = []
    for e in graph_config.edges:
        edges_config.append(_edge_format_save(e))

    # Save the internal nodes
    node_configs = {}
    node_ids = []
    for node in graph.nodes:
        node_ids.append(node.node_id)
        node_configs[node] = node.config_dict()

    if len(node_configs) > 0:
        for node in graph.nodes:
            save_node(
                node_config=node_configs[node],
                node_save_dict=node_save_dict,
                graph_save_dict=graph_save_dict,
                hp_collection=hp_collection,
                parameter_path=parameter_path,
                file_counter=file_counter,
                constant_counter=constant_counter,
                backend=node.backend,
            )

    graph_config_payload = {
        KEY_NODES_INCLUDED: node_ids,
        KEY_EDGES: edges_config,
        KEY_SORTED: graph_config.graph_was_sorted,
        KEY_FROM_OUTSIDE: [_edge_format_save(e) for e in graph_config.edges_from_outside],
        KEY_TO_OUTSIDE: [_edge_format_save(e) for e in graph_config.edges_to_outside],
    }
    if save_key in graph_save_dict:
        raise ValueError(f"Duplicate graph save_key {save_key} found in graph_save_dict.")
    graph_save_dict[save_key] = graph_config_payload


def _edge_format_save(edge: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        KEY_FROM_NODE_ID: edge[0],
        KEY_FROM_PORT: edge[1],
        KEY_TO_NODE_ID: edge[2],
        KEY_TO_PORT: edge[3],
    }
