import inspect
from collections.abc import Callable
from typing import get_args, get_origin, get_type_hints

from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import DeepLearningBackend
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.nodes import NO_DEFAULT, InputPort, Node, OutputPort


class FunctionNode(Node[TensorType]):
    """Node wrapper around an arbitrary Python callable.

    Args:
        function (Callable): The function to wrap.
        name (str | None, optional): The name of the node. If None, the
            function's __name__ attribute will be used. Defaults to None.
        backend (type[DeepLearningBackend[TensorType]], optional):
            The backend to use for the node.
    """

    def __init__(
        self,
        function: Callable,
        name: str | None = None,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.function = function
        self.function_signature = inspect.signature(function)
        try:
            self.function_type_hints = get_type_hints(function, include_extras=True)
        except (NameError, TypeError):
            self.function_type_hints = {}

        super().__init__(
            name if name is not None else getattr(function, "__name__", "FunctionNode"),
            backend=backend,
        )

        # Rebuild ports to match the wrapped function instead of the generic
        # Node.forward(*args, **kwargs) signature.
        self._build_function_ports()

    def _build_function_ports(self):
        self._input_ports = []
        for param in self.function_signature.parameters.values():
            hint = self.function_type_hints.get(param.name, param.annotation)
            config, _ = self._unwrap_annotated(hint, self)

            self._input_ports.append(
                InputPort(
                    config,
                    node=self,
                    name=param.name,
                    default=(
                        NO_DEFAULT
                        if param.default is inspect.Parameter.empty
                        else param.default
                    ),
                )
            )

        return_hint = self.function_type_hints.get(
            "return", self.function_signature.return_annotation
        )

        if return_hint is inspect.Signature.empty or return_hint is None:
            output_hints = [DataConfiguration.empty()]
        elif get_origin(return_hint) is tuple:
            output_hints = list(get_args(return_hint))
        else:
            output_hints = [return_hint]

        self._output_ports = []
        for i, output_hint in enumerate(output_hints):
            config, _ = self._unwrap_annotated(output_hint, self)
            self._output_ports.append(
                OutputPort(
                    config,
                    node=self,
                    name=f"output_{i}",
                )
            )

    def forward(self, *args, **kwargs):
        bound = self.function_signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return self.function(*bound.args, **bound.kwargs)
