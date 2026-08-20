from typing import Callable

from qewton.backends.base import Backend, TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.nodes import Node, OutputPort


class FunctionWrappingNode(Node):
    """
    A Node that wraps a plain Python callable and executes it directly
    (eagerly) at evaluation time, without tracing its execution into an
    internal graph of sub-nodes.

    In contrast to `FromFunctionNode`, which analyzes a function by calling
    it once with `TrackingObject`s (and therefore only supports operations
    understood by that tracing mechanism), `FunctionWrappingNode` treats the
    function as an opaque, atomic operation: its input ports are derived
    from the function's signature and type hints (e.g. `Variable`
    annotations), but the function body itself can be arbitrary Python
    code, since it is simply called with the real input tensors at
    evaluation time. This trades away automatic decomposition into a
    sub-graph (used e.g. for structural introspection) for full
    flexibility of the function body.

    If the function has no return type annotation, a single output port
    with an empty (dynamically resolved) `DataConfiguration` is created,
    since the output shape cannot be inferred without tracing.

    Args:
        function (Callable): The function to wrap. Called directly with
            the real input tensors at evaluation time.
        name (str, optional): The name of this node. Defaults to the
            function's `__name__`.
        backend (type[Backend[TensorType]], optional): The backend type
            for tensor operations. Defaults to Backend.
    """

    def __init__(
        self,
        function: Callable,
        name: str | None = None,
        backend: type[Backend[TensorType]] = Backend,
    ) -> None:
        self.forward = function
        super().__init__(
            name=name if name is not None else getattr(function, "__name__", None),
            backend=backend,
        )
        if len(self._output_ports) == 0:
            self._output_ports.append(
                OutputPort(DataConfiguration.empty(), node=self, name="output_0")
            )
