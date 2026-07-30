import importlib
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
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.helpers import HyperParameterState
from qewton.optim.parameters.trainable_parameters import (
    TrainableParameters,
    TrainableParametersCollection,
)
from qewton.backends import BACKEND_DICT, DEFAULT_DL_BACKEND, Backend


def _get_class(module: str, classname: str) -> type:
    """Import and return a class by its module path and name."""
    mod = importlib.import_module(module)
    return getattr(mod, classname)


def _load_hyperparameter(data: dict[str, Any]) -> HyperParameter:
    """Reconstruct a HyperParameter from its serialized dict.

    Uses the stored *class*/*module* to find the concrete subclass, then
    passes only the arguments that the constructor actually accepts so that
    subclasses with different signatures (e.g. CategoricalHyperparameter
    uses *categories* instead of *parameter_range*) are handled safely.
    """
    import inspect

    hp_class: type[HyperParameter] = _get_class(data["module"], data["class"])
    state = HyperParameterState[data["state"]]
    name = data.get("name", "")
    current_value = data.get("current_value")
    parameter_range = data.get("parameter_range")
    default_grid = data.get("default_grid", 0)

    sig = inspect.signature(hp_class.__init__)
    params = set(sig.parameters.keys()) - {"self"}

    kwargs: dict[str, Any] = {"state": state, "name": name}

    # Handle the common variants — each subclass uses different field names.
    if "categories" in params:
        # CategoricalHyperparameter / BooleanHyperparameter
        categories = (
            parameter_range if isinstance(parameter_range, list) else [current_value]
        )
        kwargs["categories"] = categories
        kwargs["initial_value"] = current_value
    elif "parameter_range" in params:
        # ContinuousHyperparameter / DiscreteHyperparameter
        kwargs["parameter_range"] = parameter_range
        kwargs["initial_value"] = current_value
    else:
        # Fallback: base HyperParameter
        kwargs["parameter_range"] = parameter_range if parameter_range is not None else []
        kwargs["initial_value"] = current_value

    if "default_grid" in params:
        kwargs["default_grid"] = default_grid

    hp: HyperParameter = hp_class(**kwargs)
    # Restore the current (possibly tuned) value.
    hp.current_value = current_value
    return hp


def _load_other_args(
    value: Any,
    root_dir: Path,
    backend_class: type,
    node_id: int,
) -> Any:
    """Recursively walk *other_args* and replace parameter paths with
    loaded *TrainableParameters* instances."""
    if isinstance(value, str) and value.startswith("trainable_parameters/"):
        abs_path = root_dir / value
        tensor = backend_class.load(abs_path)
        return TrainableParameters(node_id=node_id, parameters=tensor)
    # Constant values that are not trained:
    if isinstance(value, str) and value.startswith("constants/"):
        abs_path = root_dir / value
        return backend_class.load(abs_path)

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

    if isinstance(value, dict):
        return {
            k: _load_other_args(v, root_dir, backend_class, node_id)
            for k, v in value.items()
        }

    return value


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
    config_path = root_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {root_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        config_data: dict[str, Any] = json.load(f)

    object_type = config_data.get("object_type", None)
    if object_type == "Graph":
        return load_graph(root_dir, config_data)
    if object_type == "Node":
        node_config = load_node(root_dir, config_data)
        if node_config.node_identifier is not None:
            node_class = NODE_REGISTRY.get(node_config.node_identifier, Node)
        else:
            node_class = Node
        return node_class.load_from_config(node_config)
    raise ValueError(f"Unknown object_type '{object_type}' in config.json at {root_dir}")


def load_node(root_dir: Path, config_data: dict[str, Any]) -> NodeConfig:
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
    other_args_raw: dict[str, Any] = config_data.get("other_args", {})
    backend_ref: str = other_args_raw.get("backend", "")
    backend_class: type[Backend] = BACKEND_DICT.get(backend_ref, DEFAULT_DL_BACKEND)

    # Load hyperparameters
    hyperparameters: dict[str, HyperParameter] = {}
    hp_file = root_dir / config_data.get("hyperparameters_file", "hyperparameters.json")
    if hp_file.exists():
        with hp_file.open("r", encoding="utf-8") as f:
            hp_data: dict[str, Any] = json.load(f)
        for hp_name, hp_dict in hp_data.items():
            hyperparameters[hp_name] = _load_hyperparameter(hp_dict)

    # Reconstruct other_args
    other_args = _load_other_args(
        other_args_raw, root_dir, backend_class, config_data["node_id"]
    )
    other_args["backend"] = backend_class  # Ensure backend is set correctly

    # Rebuild NodeConfig and use load_from_config for the given node
    node_config = NodeConfig(
        node_identifier=config_data["node_identifier"],
        node_id=config_data["node_id"],
        mode=EvaluationPhase[config_data["mode"]],
        state=NodeState[config_data["state"]],
        hyperparameters=hyperparameters,
        other_args=other_args,
    )
    return node_config


def load_graph(root_dir: Path, config_data: dict[str, Any]) -> Graph:
    """Load a graph from a given path.

    Reads the *graph_config.json* from *path*, reconstructs all nodes and
    edges, and returns a fully configured *Graph* instance.

    Args:
        path (str | Path): Directory produced by *save_graph*.

    Returns:
        Graph: The reconstructed graph.
    """
    # First, load all nodes
    node_configs = {}
    for node_id in config_data["nodes_included"]:
        node_path = root_dir / f"nodes/node_{node_id}"
        node_config_path = node_path / "config.json"
        if not node_config_path.exists():
            raise FileNotFoundError(f"No config.json found for node {node_id}")
        with node_config_path.open("r", encoding="utf-8") as f:
            node_config_data: dict[str, Any] = json.load(f)
        node_configs[node_id] = load_node(node_path, node_config_data)

    # Transform edges into a list of tuples
    edges = []
    for edge in config_data["edges"]:
        edges.append(
            (
                edge["from_node_id"],
                edge["from_port"],
                edge["to_node_id"],
                edge["to_port"],
            )
        )

    # Build the config and load the graph
    graph_config = GraphConfig(
        node_configs=node_configs,
        edges=edges,
        graph_was_sorted=config_data.get("sorted", False),
    )
    return Graph.load_from_graph_config(graph_config)
