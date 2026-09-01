from enum import Enum


class SavingKeys(Enum):
    VERSION = "version"
    KEY_VERSION = 1

    KEY_TYPE = "type"
    KEY_VALUES = "values"

    KEY_SELF_ARGS = "self_args"
    KEY_MODULE = "module"
    KEY_CLASS = "class"

    KEY_LIST = "list"
    KEY_TUPLE = "tuple"
    KEY_DICT = "dict"
    KEY_SET = "set"
    KEY_SERIALIZABLE = "serializable"
    KEY_CLASS = "class"
    KEY_MODULE = "module"

    NODE_ID = "node_id"
    NODE_MODE = "node_mode"
    NODE_STATE = "node_state"

    FILE_CONFIG = "config"
    FILE_DATA = "data"
    FILE_PARAMETERS = "parameters"
