class DataConfigMismatchError(ValueError):
    """Raised when two data configurations that are expected
    to match do not match or when they are tried to be unified."""


class StructureUnknown(Exception):
    """The structure of a configuration is not known far enough for a shape
    operation yet, e.g. the position of a dim behind an ellipsis.

    Raised by the shape operations of a configuration while a graph is
    connected; the graph tries the operation again once more is known.
    """
