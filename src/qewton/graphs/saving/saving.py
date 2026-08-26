from __future__ import annotations

import json
import os
import logging
import shutil
from pathlib import Path
from typing import Any

from qewton.graphs.nodes import Node, NodeConfig
from qewton.graphs.graphs import Graph
from qewton.backends import Backend
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.saving.hyperparameter_saving import (
    serialize_hyperparameter,
    _add_hp_to_collection,
)
from qewton.graphs.saving.serialize_objects import (
    SaveContext,
    _serialize_other_args,
    serialize_port,
)
from qewton.graphs.saving.serialize_dataconfigurations import (
    serialize_data_configurations,
    DataConfigSerializationResult,
)

from qewton.graphs.saving.schema import (
    DIR_PARAMETERS,
    FILE_NODES,
    FILE_GRAPHS,
    FILE_CONFIG,
    FILE_HYPERPARAMETERS,
    KEY_EDGES,
    KEY_FROM_NODE_ID,
    KEY_FROM_OUTSIDE,
    KEY_FROM_PORT,
    KEY_HYPERPARAMETERS,
    KEY_DATA_CONFIGURATIONS,
    KEY_MODE,
    KEY_NODE_ID,
    KEY_NODE_IDENTIFIER,
    KEY_NODES_INCLUDED,
    KEY_OBJECT_TYPE,
    KEY_OTHER_ARGS,
    KEY_NESTED_GRAPHS,
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
    TYPE_TUPLE,
    KEY_INPUT_PORTS,
    KEY_OUTPUT_PORTS,
)

logger = logging.getLogger(__name__)


def save(obj: Node | Graph, path: str | Path, replace: bool = False) -> None:
    """Saves a Node or Graph to a file.

    Args:
        obj (Node | Graph): The Node or Graph to save.
        path (str): The path to the file where the object will be saved.
    """
    # Save path and create a temporary one to save the new data object
    original_path = Path(path)
    path = Path(str(original_path) + "_temp_save")

    if Path(original_path).exists() and not replace:
        raise FileExistsError(
            f"The path {original_path} already exists. Use replace=True to allow to overwrite it."
        )

    logger.info("Saving %s to %s", obj.__class__.__name__, path)

    # Setup all the necessary directories and counters for saving parameters and constants
    save_context = SaveContext(
        parameters_dir=Path(path) / DIR_PARAMETERS,
        file_counter=0,
        constants_file_counter=0,
        node_save_dict={},
        graph_save_dict={},
        hp_collection={},
        input_port_map={},
        output_port_map={},
        data_cdf_seri=DataConfigSerializationResult({}, [], [], []),
    )

    config_dict: dict = {KEY_SCHEMA_VERSION: SCHEMA_VERSION}
    if not save_context.parameters_dir.exists():
        save_context.parameters_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, Node):
        config_dict[KEY_OBJECT_TYPE] = OBJECT_TYPE_NODE
        config_dict[KEY_NODE_ID] = obj.node_id
        save_node(obj.__getstate__(), save_context, obj.backend)
    elif isinstance(obj, Graph):
        config_dict[KEY_OBJECT_TYPE] = OBJECT_TYPE_GRAPH
        save_graph(graph=obj, save_key=OBJECT_TYPE_GRAPH, save_context=save_context)
    else:
        raise TypeError(f"Object of type {type(obj)} is not supported for saving.")
    # Save the node and graph configurations to JSON files

    with open(Path(path) / FILE_CONFIG, "w", encoding="utf-8") as f:
        f.write(json.dumps(config_dict, indent=4))
    with open(Path(path) / FILE_NODES, "w", encoding="utf-8") as f:
        f.write(json.dumps(save_context.node_save_dict, indent=4))
    with open(Path(path) / FILE_GRAPHS, "w", encoding="utf-8") as f:
        f.write(json.dumps(save_context.graph_save_dict, indent=4))
    # Save the hyperparameter collection to a JSON file
    with open(Path(path) / FILE_HYPERPARAMETERS, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    k: serialize_hyperparameter(v)
                    for k, v in save_context.hp_collection.items()
                },
                indent=4,
            )
        )
    # The parameters and constants are saved in their respective files during the
    # serialization process.

    if Path(original_path).exists():
        shutil.rmtree(original_path)
    # rename the temporary save directory to the original path
    os.rename(path, original_path)

    logger.info("Saving completed")


def save_node(
    node_config: NodeConfig,
    save_context: SaveContext,
    backend: type[Backend],
):
    """Saves a Node to a file."""
    if node_config.node_id in save_context.node_save_dict:
        return  # Node already saved, skip to avoid duplicates

    # Serialize the node configuration and save it to a JSON file
    config_payload = {
        KEY_NODE_IDENTIFIER: node_config.node_identifier,
        KEY_NODE_ID: node_config.node_id,
        KEY_MODE: node_config.mode.name,
        KEY_STATE: node_config.state.name,
    }
    # Collect all data configurations from the node's ports and add them to
    # the config payload
    cfg_serialization = serialize_data_configurations(node_config)
    config_payload[KEY_DATA_CONFIGURATIONS] = cfg_serialization.serialized_payload
    save_context.data_cdf_seri = cfg_serialization
    # Add ports to the config payload
    save_context.input_port_map = {}
    save_context.output_port_map = {}
    if node_config.input_ports:
        config_payload[KEY_INPUT_PORTS] = [
            serialize_port(p, i, save_context)
            for i, p in enumerate(node_config.input_ports)
        ]
    if node_config.output_ports:
        config_payload[KEY_OUTPUT_PORTS] = [
            serialize_port(p, i, save_context)
            for i, p in enumerate(node_config.output_ports)
        ]
    # Split arguments into hyperparameters, nested graphs, and other arguments
    other_args = {}
    hyperparameters = {}
    nested_graphs = {}
    for k, v in node_config.self_args.items():
        if isinstance(v, HyperParameter):
            hyperparameters[k] = v
        elif (
            isinstance(v, (list, tuple))
            and len(v) > 0
            and all(isinstance(i, HyperParameter) for i in v)
        ):
            hyperparameters[k] = v
        elif (
            isinstance(v, dict)
            and len(v) > 0
            and all(isinstance(i, HyperParameter) for i in v.values())
        ):
            hyperparameters[k] = v
        elif isinstance(v, Graph):
            nested_graphs[k] = v
        else:
            other_args[k] = v
    # Serialize arguments, including trainable parameters,
    # and save them to the main config
    node_dependencies: list[Node] = []
    serialized_other_args = _serialize_other_args(
        other_args,
        save_context=save_context,
        node_dependencies=node_dependencies,
        backend=backend,
    )
    config_payload[KEY_OTHER_ARGS] = serialized_other_args

    # check for nested graphs inside the node:
    if len(nested_graphs) > 0:
        config_payload[KEY_NESTED_GRAPHS] = {}
        for graph_name, graph in nested_graphs.items():
            save_name = f"{graph_name}_id_{node_config.node_id}"
            save_graph(graph=graph, save_key=save_name, save_context=save_context)
            config_payload[KEY_NESTED_GRAPHS][graph_name] = save_name

    # save all other nodes that are dependencies of this node
    for dep_node in node_dependencies:
        save_node(
            node_config=dep_node.__getstate__(),
            save_context=save_context,
            backend=dep_node.backend,
        )

    # Save the node configuration to the node_save_dict, ensuring no duplicate node_ids
    save_context.node_save_dict[node_config.node_id] = config_payload

    # Hyperparameters are saved on the global level, so we dont have duplicates
    # just add them to the collection
    if len(hyperparameters) > 0:
        _hp_mapping(node_config, save_context, config_payload, hyperparameters)


def save_graph(
    graph: Graph,
    save_key: str,
    save_context: SaveContext,
) -> None:
    """Saves a Graph to a file.

    Args:
        graph (Graph): The Graph to save.
        path (str): The path to the file where the Graph will be saved.
    """
    # Serialize the graph configuration and save it to a JSON file
    graph_config = graph.__getstate__()

    # Improve edge readability by converting node objects to their IDs
    edges_config = []
    for e in graph_config.edges:
        edges_config.append(_edge_format_save(e))

    # Save the internal nodes
    node_configs = {}
    node_ids = []
    for node in graph.nodes:
        node_ids.append(node.node_id)
        node_configs[node] = node.__getstate__()

    if len(node_configs) > 0:
        for node in graph.nodes:
            save_node(
                node_config=node_configs[node],
                save_context=save_context,
                backend=node.backend,
            )

    graph_config_payload = {
        KEY_NODES_INCLUDED: node_ids,
        KEY_EDGES: edges_config,
        KEY_SORTED: graph_config.graph_was_sorted,
        KEY_FROM_OUTSIDE: [_edge_format_save(e) for e in graph_config.edges_from_outside],
        KEY_TO_OUTSIDE: [_edge_format_save(e) for e in graph_config.edges_to_outside],
    }
    if save_key in save_context.graph_save_dict:
        raise ValueError(f"Duplicate graph save_key {save_key} found in graph_save_dict.")
    save_context.graph_save_dict[save_key] = graph_config_payload


def _edge_format_save(edge: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        KEY_FROM_NODE_ID: edge[0],
        KEY_FROM_PORT: edge[1],
        KEY_TO_NODE_ID: edge[2],
        KEY_TO_PORT: edge[3],
    }


def _hp_mapping(node_config, save_context, config_payload, hyperparameters):
    for hp in hyperparameters.values():
        if isinstance(hp, (list, tuple)):
            for sub_hp in hp:
                _add_hp_to_collection(
                    sub_hp, save_context.hp_collection, node_config.node_id
                )
        else:
            _add_hp_to_collection(hp, save_context.hp_collection, node_config.node_id)
        # Build the reference:
    config_payload[KEY_HYPERPARAMETERS] = {}
    for name, hp in hyperparameters.items():
        if isinstance(hp, list):
            config_payload[KEY_HYPERPARAMETERS][name] = [sub_hp.name for sub_hp in hp]
        elif isinstance(hp, tuple):
            config_payload[KEY_HYPERPARAMETERS][name] = {
                KEY_TYPE: TYPE_TUPLE,
                KEY_VALUES: [sub_hp.name for sub_hp in hp],
            }
        else:
            config_payload[KEY_HYPERPARAMETERS][name] = hp.name
