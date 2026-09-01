from enum import Enum


class SavingKeys(Enum):
    KEY_TYPE = "type"
    KEY_VALUES = "values"

    KEY_LIST = "list"
    KEY_TUPLE = "tuple"
    KEY_DICT = "dict"
    KEY_SET = "set"
    KEY_NODE = "node"

    NODE_IDENTIFIER = "node_identifier"
    NODE_ID = "node_id"
    NODE_MODE = "node_mode"
    NODE_STATE = "node_state"
    NODE_SELF_ARGS = "node_self_args"
