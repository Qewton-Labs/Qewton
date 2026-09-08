from typing import Annotated

from qewton.backends import DEFAULT_DL_BACKEND, TensorType
from qewton.backends.base import Backend
from qewton.graphs.nodes import Node, NodeState
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import EllipsisAxes


class Identity(Node[TensorType]):
    """A node that just returns the provided input. Helpful to
    track some operations and and inputs.
    """

    ell_axis = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ell_axis)]
    ) -> Annotated[TensorType, DC(ell_axis)]:
        return x


class Zeros(Node[TensorType]):

    def __init__(
        self,
        shape: int | tuple[int, ...],
        name: str | None = None,
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        super().__init__(name, state, backend)

    def forward(self) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.zeros(self.shape)


class ZerosLike(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.zeros_like(x)


class Ones(Node[TensorType]):

    def __init__(
        self,
        shape: int | tuple[int, ...],
        name: str | None = None,
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        super().__init__(name, state, backend)

    def forward(self) -> Annotated[TensorType, DC(EllipsisAxes())]:
        return self.backend.math.ones(self.shape)


class OnesLike(Node[TensorType]):
    ellipsis_dims = EllipsisAxes()

    def forward(
        self, x: Annotated[TensorType, DC(ellipsis_dims)]
    ) -> Annotated[TensorType, DC(ellipsis_dims)]:
        return self.backend.math.ones_like(x)
