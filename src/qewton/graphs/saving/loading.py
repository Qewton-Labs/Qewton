import importlib
import inspect
import json
from pathlib import Path
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

from qewton.graphs.saving.hyperparameter_codec import (
    deserialize_hyperparameter,
    _inverse_hp_dict,
)
from qewton.geometries.base import GEOMETRY_REGISTRY
from qewton.graphs.saving.schema import (
    DIR_NODES,
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
    KEY_MODULE,
    KEY_NESTED_GRAPHS,
    KEY_NODE_ID,
    KEY_NODE_IDENTIFIER,
    KEY_NODES_INCLUDED,
    KEY_OBJECT_TYPE,
    KEY_OTHER_ARGS,
    KEY_PATH,
    KEY_SORTED,
    KEY_STATE,
    KEY_TO_NODE_ID,
    KEY_TO_OUTSIDE,
    KEY_TO_PORT,
    KEY_TYPE,
    KEY_VALUES,
    OBJECT_TYPE_GRAPH,
    OBJECT_TYPE_NODE,
    TYPE_BACKEND_REF,
    TYPE_CLASS_REF,
    TYPE_CONSTANT_REF,
    TYPE_HP_REF,
    TYPE_SET,
    TYPE_TRAINABLE_PARAMETER_REF,
    TYPE_TUPLE,
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

    # Read config.json
    config_path = root_dir / FILE_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"No {FILE_CONFIG} found in {root_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        config_data: dict[str, Any] = json.load(f)

    object_type = config_data.get(KEY_OBJECT_TYPE, None)
    if object_type == OBJECT_TYPE_GRAPH:
        graph_config = load_graph(root_dir, config_data)
        return Graph.load_from_graph_config(graph_config)
    if object_type == OBJECT_TYPE_NODE:
        node_config = load_node(root_dir, config_data)
        if node_config.node_identifier is not None:
            node_class = NODE_REGISTRY.get(node_config.node_identifier, Node)
        else:
            node_class = Node
        return node_class.load_from_config(node_config)
    raise ValueError(f"Unknown object_type '{object_type}' in config.json at {root_dir}")


def load_node(
    root_dir: Path,
    config_data: dict[str, Any],
    shared_hps: dict[str, Any] | None = None,
) -> NodeConfig:
    """Load a node from a given path.

    Reads the *config.json* (and *hyperparameters.json* when present) from
    *path*, reconstructs all hyperparameters and trainable parameters, the
    method returns a fully configured *NodeConfig* instance. The caller can then
    returns a fully configured *Node* instance via the existing
    *Node.load_from_config* method.

    Args:
        path (str | Path): Directory produced by *save_node*.

    Returns:
        NodeConfig: The reconstructed NodeConfig.
    """
    # Resolve the backend, since we need it to load the tensors
    other_args_raw: dict[str, Any] = config_data.get(KEY_OTHER_ARGS, {})
    backend_class = _resolve_backend_class(other_args_raw.get(KEY_BACKEND_KEY))

    # Load hyperparameters
    if shared_hps is None:
        shared_hps = _load_hyperparameters(root_dir)

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

    # Check for nested graphs and construct them
    nested_graphs = {}
    if KEY_NESTED_GRAPHS in config_data:
        for graph_name, graph_path in config_data[KEY_NESTED_GRAPHS].items():
            graph_root_dir = root_dir / graph_path
            graph_json_path = graph_root_dir / FILE_CONFIG
            with graph_json_path.open("r", encoding="utf-8") as f:
                graph_config_data: dict[str, Any] = json.load(f)
            nested_graphs[graph_name] = load_graph(
                graph_root_dir, graph_config_data, shared_hps=shared_hps
            )

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


def load_graph(
    root_dir: Path, config_data: dict[str, Any], shared_hps: dict[str, Any] | None = None
) -> GraphConfig:

    # Load hyperparameters
    if shared_hps is None:
        shared_hps = _load_hyperparameters(root_dir)

    # First, load all nodes
    node_configs = {}
    for node_id in config_data[KEY_NODES_INCLUDED]:
        node_path = root_dir / f"{DIR_NODES}/node_{node_id}"
        node_config_path = node_path / FILE_CONFIG
        if not node_config_path.exists():
            raise FileNotFoundError(f"No {FILE_CONFIG} found for node {node_id}")
        with node_config_path.open("r", encoding="utf-8") as f:
            node_config_data: dict[str, Any] = json.load(f)
        node_configs[node_id] = load_node(
            node_path, node_config_data, shared_hps=shared_hps
        )

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
        node_configs=node_configs,
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
