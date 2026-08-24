import importlib
import inspect
import json
from pathlib import Path
from collections import deque
from typing import Any

from qewton.graphs.nodes import (
    Node,
    NodeConfig,
    NodeState,
    EvaluationPhase,
    NODE_REGISTRY,
)
from qewton.graphs.graphs import Graph, GraphConfig
from qewton.optim.parameters.trainable_parameters import (
    TrainableParameters,
    TrainableParametersCollection,
)
from qewton.backends import BACKEND_DICT, DEFAULT_DL_BACKEND, Backend
from qewton.config.variables import Variable

from qewton.graphs.saving.hyperparameter_saving import (
    deserialize_hyperparameter,
    _inverse_hp_dict,
)
from qewton.geometries.base import GEOMETRY_REGISTRY
from qewton.graphs.saving.schema import (
    OBJECT_TYPE_NODE,
    OBJECT_TYPE_GRAPH,
    KEY_OBJECT_TYPE,
    FILE_CONFIG,
    FILE_NODES,
    FILE_GRAPHS,
    FILE_HYPERPARAMETERS,
    KEY_BACKEND_KEY,
    KEY_CLASS,
    KEY_EDGES,
    KEY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    KEY_FROM_NODE_ID,
    KEY_FROM_OUTSIDE,
    KEY_FROM_PORT,
    KEY_HP_KEY,
    KEY_HYPERPARAMETERS,
    KEY_MODE,
    KEY_MODULE,
    KEY_NESTED_GRAPHS,
    KEY_NODE_ID,
    KEY_NODE_IDENTIFIER,
    KEY_NODES_INCLUDED,
    KEY_OTHER_ARGS,
    KEY_PATH,
    KEY_SORTED,
    KEY_STATE,
    KEY_TO_NODE_ID,
    KEY_TO_OUTSIDE,
    KEY_TO_PORT,
    KEY_TYPE,
    KEY_VALUES,
    TYPE_BACKEND_REF,
    TYPE_CLASS_REF,
    TYPE_CONSTANT_REF,
    TYPE_HP_REF,
    TYPE_SET,
    TYPE_TRAINABLE_PARAMETER_REF,
    TYPE_TUPLE,
    TYPE_NODE,
    TYPE_VARIABLE,
    TYPE_GEOMETRY,
    TYPE_ELLIPSIS,
)

_ALLOWED_MODULE_PREFIXES = ("qewton.",)


def _get_class(module: str, classname: str) -> type:
    """Import and return a class by its module path and name.

    Only modules from trusted package prefixes are allowed to prevent
    arbitrary code execution via crafted save files.
    """
    if not any(module.startswith(prefix) for prefix in _ALLOWED_MODULE_PREFIXES):
        raise ValueError(
            f"Refusing to import '{module}.{classname}': module is not in the "
            f"allowed prefixes {_ALLOWED_MODULE_PREFIXES}. If you need to load "
            f"classes from another package, add its prefix to _ALLOWED_MODULE_PREFIXES."
        )
    mod = importlib.import_module(module)
    return getattr(mod, classname)


def _load_hyperparameter_entry(value: Any) -> Any:
    if isinstance(value, dict):
        value_type = value.get(KEY_TYPE)
        if value_type == TYPE_TUPLE:
            return tuple(_load_hyperparameter_entry(v) for v in value[KEY_VALUES])
        if value_type == TYPE_SET:
            return set(_load_hyperparameter_entry(v) for v in value[KEY_VALUES])
        if KEY_CLASS in value and value[KEY_CLASS] in _inverse_hp_dict:
            return deserialize_hyperparameter(value)
    if isinstance(value, list):
        return [_load_hyperparameter_entry(v) for v in value]
    raise ValueError(f"Unexpected hyperparameter payload: {value}")


def _resolve_hp_entry(value: Any, shared_hps: dict[str, Any]) -> Any:
    """Resolve a hyperparameter entry, following hp_ref pointers into shared_hps."""
    if isinstance(value, dict):
        value_type = value.get(KEY_TYPE)
        if value_type == TYPE_HP_REF:
            return shared_hps[value[KEY_HP_KEY]]
        if value_type == TYPE_TUPLE:
            return tuple(_resolve_hp_entry(v, shared_hps) for v in value[KEY_VALUES])
        if value_type == TYPE_SET:
            return set(_resolve_hp_entry(v, shared_hps) for v in value[KEY_VALUES])
    if isinstance(value, list):
        return [_resolve_hp_entry(v, shared_hps) for v in value]
    return _load_hyperparameter_entry(value)


def _resolve_backend_class(value: Any) -> type[Backend]:
    if isinstance(value, dict):
        value_type = value.get(KEY_TYPE)
        if value_type == TYPE_BACKEND_REF:
            backend_key = value.get(KEY_BACKEND_KEY)
            if isinstance(backend_key, str):
                return BACKEND_DICT.get(backend_key, DEFAULT_DL_BACKEND)
            return DEFAULT_DL_BACKEND
        if value_type == TYPE_CLASS_REF:
            maybe_cls = _get_class(value[KEY_MODULE], value[KEY_CLASS])
            if inspect.isclass(maybe_cls) and issubclass(maybe_cls, Backend):
                return maybe_cls
    if isinstance(value, str):
        return BACKEND_DICT.get(value, DEFAULT_DL_BACKEND)
    return DEFAULT_DL_BACKEND


def _load_other_args(
    value: Any,
    root_dir: Path,
    backend_class: type,
    node_id: int,
) -> Any:
    """Recursively walk *other_args* and replace parameter paths with
    loaded *TrainableParameters* instances."""
    if isinstance(value, dict):
        value_type = value.get(KEY_TYPE)
        if value_type == TYPE_ELLIPSIS:
            return ...
        if value_type == TYPE_GEOMETRY:
            inner = {
                k: _load_other_args(v, root_dir, backend_class, node_id)
                for k, v in value[KEY_VALUES].items()
            }
            cls_name = inner.get("class", "")
            cls_obj = GEOMETRY_REGISTRY.get(cls_name)
            if cls_obj is None:
                raise ValueError(
                    f"Unknown geometry class '{cls_name}' in saved configuration. "
                    f"Ensure that the class is registered in GEOMETRY_REGISTRY."
                )
            return cls_obj.load(dict(inner))
        if value_type == TYPE_VARIABLE:
            return Variable.from_dict(value[KEY_VALUES])
        if value_type == TYPE_TRAINABLE_PARAMETER_REF:
            abs_path = root_dir / value[KEY_PATH]
            tensor = backend_class.load(abs_path)
            return TrainableParameters(node_id=node_id, parameters=tensor)
        if value_type == TYPE_CONSTANT_REF:
            abs_path = root_dir / value[KEY_PATH]
            return backend_class.load(abs_path)
        if value_type == TYPE_TUPLE:
            return tuple(
                _load_other_args(v, root_dir, backend_class, node_id)
                for v in value[KEY_VALUES]
            )
        if value_type == TYPE_SET:
            return set(
                _load_other_args(v, root_dir, backend_class, node_id)
                for v in value[KEY_VALUES]
            )
        if value_type == TYPE_CLASS_REF:
            return _get_class(value[KEY_MODULE], value[KEY_CLASS])
        if value_type == TYPE_BACKEND_REF:
            backend_key = value.get(KEY_BACKEND_KEY)
            if isinstance(backend_key, str):
                return BACKEND_DICT.get(backend_key, DEFAULT_DL_BACKEND)
            return DEFAULT_DL_BACKEND
        if value_type == TYPE_NODE:
            return value[KEY_NODE_ID]
        return {
            k: _load_other_args(v, root_dir, backend_class, node_id)
            for k, v in value.items()
        }

    if isinstance(value, list):
        # Could be a list of parameter paths (multiple groups)
        loaded = [_load_other_args(v, root_dir, backend_class, node_id) for v in value]
        # If every item became a TrainableParameters, combine them
        if all(isinstance(v, TrainableParameters) for v in loaded):
            collection = TrainableParametersCollection()
            for tp in loaded:
                collection.add(tp)
            return collection
        return loaded

    return value


def _load_hyperparameters(root_dir):
    shared_hps = {}
    hp_file = Path(root_dir / FILE_HYPERPARAMETERS)
    if hp_file.exists():
        with hp_file.open("r", encoding="utf-8") as f:
            hp_data: dict[str, Any] = json.load(f)
        for hp_name, hp_value in hp_data.items():
            shared_hps[hp_name] = _load_hyperparameter_entry(hp_value)
    return shared_hps


#########################################################################
def load(path: str | Path) -> Node | Graph:
    """Load a node or graph from a given path.

    Args:
        path (str | Path): Directory produced by *qewton.save*.

    Returns:
        Node | Graph: The reconstructed node or graph.
    """
    root_dir = Path(path)

    # Check and read all json files
    files = [
        root_dir / FILE_CONFIG,
        root_dir / FILE_NODES,
        root_dir / FILE_GRAPHS,
        root_dir / FILE_HYPERPARAMETERS,
    ]
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"No {file} found in {root_dir}")

    # Check for version mismatch
    with files[0].open("r", encoding="utf-8") as f:
        config_data: dict[str, Any] = json.load(f)
    assert config_data.get(KEY_SCHEMA_VERSION) == SCHEMA_VERSION, (
        f"Schema version mismatch: expected {SCHEMA_VERSION}, "
        f"found {config_data.get(KEY_SCHEMA_VERSION)} in {files[0]}"
    )
    # Load all hyperparameters
    shared_hps = _load_hyperparameters(root_dir)
    # Load the node data and graph data
    loaded_nodes: dict[int, Node] = {}
    with files[1].open("r", encoding="utf-8") as f:
        node_data: dict[str, Any] = json.load(f)
    with files[2].open("r", encoding="utf-8") as f:
        graph_data: dict[str, Any] = json.load(f)

    # Now loop over all nodes and load them, since they can depend on each other,
    # we need to do this in a loop until all nodes are loaded.
    unloaded_queue = deque(node_data.keys())
    while unloaded_queue:
        node_id = unloaded_queue.popleft()
        loaded_node_config = load_node(
            root_dir,
            node_data[node_id],
            shared_hps=shared_hps,
            graph_data=graph_data,
            loaded_nodes=loaded_nodes,
        )
        if loaded_node_config is None:
            unloaded_queue.append(node_id)
            if len(unloaded_queue) == 1:
                raise ValueError(
                    f"Could not load node {node_id} due to unresolved dependencies."
                )
        else:
            if loaded_node_config.node_identifier is not None:
                node_class = NODE_REGISTRY.get(loaded_node_config.node_identifier, Node)
            else:
                node_class = Node
            loaded_nodes[int(node_id)] = node_class.load_from_config(loaded_node_config)

    # Now check if we should load a graph or a single node
    if config_data[KEY_OBJECT_TYPE] == OBJECT_TYPE_NODE:
        return loaded_nodes[config_data[KEY_NODE_ID]]
    if config_data[KEY_OBJECT_TYPE] == OBJECT_TYPE_GRAPH:
        main_graph_config = load_graph(graph_data[OBJECT_TYPE_GRAPH], loaded_nodes)
        return Graph.load_from_graph_config(main_graph_config)
    raise ValueError(
        f"Unknown object type '{config_data[KEY_OBJECT_TYPE]}' in {files[0]}"
    )


def load_node(
    root_dir: Path,
    config_data: dict[str, Any],
    shared_hps: dict[str, Any],
    graph_data: dict[str, Any],
    loaded_nodes: dict[int, Node],
) -> NodeConfig | None:
    """Load a node from a given path."""
    # First check if there any node dependencies that are not yet loaded,
    # if so, we return None
    for args in config_data.get(KEY_OTHER_ARGS, {}).values():
        if isinstance(args, dict) and args.get(KEY_TYPE) == TYPE_NODE:
            node_id = args.get(KEY_NODE_ID)
            if node_id not in loaded_nodes:
                return None
            args[KEY_NODE_ID] = loaded_nodes[node_id]

    # Second check if there are nested graphs, and if so, can we load them:
    nested_graphs = {}
    if KEY_NESTED_GRAPHS in config_data:
        # First check if they can be loaded
        for saved_graph_names in config_data[KEY_NESTED_GRAPHS].values():
            for node_ids in graph_data[saved_graph_names][KEY_NODES_INCLUDED]:
                if node_ids not in loaded_nodes:
                    return None
        # Now we load them
        for graph_name, saved_graph_names in config_data[KEY_NESTED_GRAPHS].items():
            nested_graphs[graph_name] = load_graph(
                graph_data[saved_graph_names], loaded_nodes
            )

    # Resolve the backend, since we need it to load the tensors
    other_args_raw: dict[str, Any] = config_data.get(KEY_OTHER_ARGS, {})
    backend_class = _resolve_backend_class(other_args_raw.get(KEY_BACKEND_KEY))

    # Reconstruct other_args
    other_args = _load_other_args(
        other_args_raw, root_dir, backend_class, config_data[KEY_NODE_ID]
    )
    # Resolve hyperparameters, following any references into shared_hps
    hyperparameters = {}
    for name, hp_name in config_data.get(KEY_HYPERPARAMETERS, {}).items():
        if isinstance(hp_name, str) and hp_name in shared_hps:
            hyperparameters[name] = shared_hps[hp_name]
        elif isinstance(hp_name, list):
            hyperparameters[name] = [shared_hps[n] for n in hp_name]
        elif isinstance(hp_name, dict) and hp_name.get(KEY_TYPE) == TYPE_TUPLE:
            hyperparameters[name] = tuple(shared_hps[n] for n in hp_name[KEY_VALUES])

    # Rebuild NodeConfig and use load_from_config for the given node
    node_config = NodeConfig(
        node_identifier=config_data[KEY_NODE_IDENTIFIER],
        node_id=config_data[KEY_NODE_ID],
        mode=EvaluationPhase[config_data[KEY_MODE]],
        state=NodeState[config_data[KEY_STATE]],
        hyperparameters=hyperparameters,
        other_args=other_args,
        nested_graphs=nested_graphs,
    )
    return node_config


def load_graph(config_data: dict[str, Any], loaded_nodes: dict[int, Node]) -> GraphConfig:

    # First read all the needed does
    nodes = {
        node_id: loaded_nodes[node_id] for node_id in config_data[KEY_NODES_INCLUDED]
    }

    # Transform edges into a list of tuples
    edges = []
    for edge in config_data[KEY_EDGES]:
        edges.append(_edge_format_to_tuple(edge))
    # Transform edges from outside into a list of tuples
    edges_from_outside = []
    for edge in config_data.get(KEY_FROM_OUTSIDE, []):
        edges_from_outside.append(_edge_format_to_tuple(edge))
    # Transform edges to outside into a list of tuples
    edges_to_outside = []
    for edge in config_data.get(KEY_TO_OUTSIDE, []):
        edges_to_outside.append(_edge_format_to_tuple(edge))

    # Build the config
    return GraphConfig(
        nodes=nodes,
        edges=edges,
        graph_was_sorted=config_data.get(KEY_SORTED, False),
        edges_from_outside=edges_from_outside,
        edges_to_outside=edges_to_outside,
    )


def _edge_format_to_tuple(edge):
    return (
        edge[KEY_FROM_NODE_ID],
        edge[KEY_FROM_PORT],
        edge[KEY_TO_NODE_ID],
        edge[KEY_TO_PORT],
    )
