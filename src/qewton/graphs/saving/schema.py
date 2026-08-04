"""Shared serialization schema constants for graph/node save/load."""

SCHEMA_VERSION = 1

# Top-level object keys
KEY_SCHEMA_VERSION = "schema_version"
KEY_OBJECT_TYPE = "object_type"

OBJECT_TYPE_NODE = "Node"
OBJECT_TYPE_GRAPH = "Graph"

# Common payload keys
KEY_CLASS = "class"
KEY_MODULE = "module"
KEY_NAME = "name"
KEY_STATE = "state"
KEY_TYPE = "type"
KEY_VALUES = "values"

# Node config keys
KEY_NODE_IDENTIFIER = "node_identifier"
KEY_NODE_ID = "node_id"
KEY_MODE = "mode"
KEY_OTHER_ARGS = "other_args"
KEY_HYPERPARAMETERS_FILE = "hyperparameters_file"
KEY_NESTED_GRAPHS = "nested_graphs"

# Graph config keys
KEY_NODES_INCLUDED = "nodes_included"
KEY_EDGES = "edges"
KEY_SORTED = "sorted"
KEY_FROM_OUTSIDE = "from_outside"
KEY_TO_OUTSIDE = "to_outside"

# Edge keys
KEY_FROM_NODE_ID = "from_node_id"
KEY_FROM_PORT = "from_port"
KEY_TO_NODE_ID = "to_node_id"
KEY_TO_PORT = "to_port"

# Directories/files
FILE_CONFIG = "config.json"
FILE_HYPERPARAMETERS = "hyperparameters.json"
DIR_NODES = "nodes"
DIR_NESTED_GRAPHS = "nested_graphs"
DIR_TRAINABLE_PARAMETERS = "trainable_parameters"
DIR_CONSTANTS = "constants"

# Hyperparameter payload keys
KEY_PARAMETER_RANGE = "parameter_range"
KEY_CURRENT_VALUE = "current_value"
KEY_DEFAULT_GRID = "default_grid"
KEY_CONDITION = "condition"
KEY_EXTRA_ARGS = "extra_args"

KEY_BOOLEAN_HP = "boolean_hyperparameter"
KEY_CONTINUOUS_HP = "continuous_hyperparameter"
KEY_DISCRETE_HP = "discrete_hyperparameter"
KEY_CATEGORICAL_HP = "categorical_hyperparameter"

# Typed JSON wrappers
TYPE_TUPLE = "tuple"
TYPE_SET = "set"
TYPE_CLASS_REF = "class_ref"
TYPE_ENUM_REF = "enum_ref"
TYPE_BACKEND_REF = "backend_ref"
TYPE_TRAINABLE_PARAMETER_REF = "trainable_parameter_ref"
TYPE_CONSTANT_REF = "constant_ref"

# Typed-ref specific keys
KEY_REF_KIND = "ref_kind"
KEY_BACKEND_KEY = "backend_key"
KEY_PATH = "path"

# Shared hyperparameter ref (used when saving nodes inside a graph)
TYPE_HP_REF = "hp_ref"
KEY_HP_KEY = "hp_key"
KEY_HYPERPARAMETERS = "hyperparameters"
