"""Positional embeddings."""

from typing import Annotated

from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.graphs.nodes import Node


class GridEmbeddingND(Node[TensorType]):
    """Positional embedding that appends the coordinates of a regular grid as channels.

    An input of shape `(batch, in_channels, x_1, ..., x_dim)` is mapped to
    `(batch, in_channels + dim, x_1, ..., x_dim)`. The coordinates along dimension `i`
    are `linspace(start, stop, x_i + 1)[:-1]`, i.e. the right boundary is excluded
    (periodic domain).

    Attributes:
        out_channels (int): Number of channels of the output, `in_channels + dim`.

    Args:
        in_channels (int): Number of channels of the input.
        dim (int, optional): Number of spatial dimensions. Defaults to 2.
        grid_boundaries (tuple[tuple[float, float], ...] | None, optional): The
            `(start, stop)` of the grid along every dimension. Defaults to the unit
            cube, `((0, 1),) * dim`.
        name (str, optional): Name of the node. Defaults to "grid_embedding".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.

    Raises:
        ValueError: If `grid_boundaries` does not have one entry per dimension, or if
            the input of `forward` has a different number of dimensions or channels.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 2,
        grid_boundaries: tuple[tuple[float, float], ...] | None = None,
        name: str = "grid_embedding",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        if grid_boundaries is None:
            grid_boundaries = ((0, 1),) * dim
        if len(grid_boundaries) != dim:
            raise ValueError(
                f"Expected grid_boundaries of length {dim}, received {grid_boundaries}."
            )
        self.in_channels = in_channels
        self.dim = dim
        self.grid_boundaries = grid_boundaries

        # Shape information for the data configurations
        self.batch_axes = BatchAxes(AxesDim(None))
        self.geo_axes = GeometryAxes(shape=tuple(AxesDim(None) for _ in range(dim)))
        super().__init__(name, backend=backend)
        self.backend: type[Backend[TensorType]] = backend

        # the grid only depends on the boundaries and on the spatial shape, dtype and
        # device of the input
        self._grid_key = None
        self._grid = None

    @property
    def out_channels(self) -> int:
        """Number of output channels: the input channels plus one per dimension."""
        return self.in_channels + self.dim

    def x_data_config(self) -> DC:
        return DC(
            self.batch_axes,
            FeatureAxes(shape=(self.in_channels,)),
            self.geo_axes,
            dtype=self.backend.default_dtype,
        )

    def out_data_config(self) -> DC:
        return DC(
            self.batch_axes,
            FeatureAxes(shape=(self.out_channels,)),
            self.geo_axes,
            dtype=self.backend.default_dtype,
        )

    def forward(
        self, data: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        """Appends the grid coordinates to the channels of the input.

        Args:
            data (TensorType): Input of shape `(batch, in_channels, x_1, ..., x_dim)`.

        Returns:
            TensorType: Tensor of shape `(batch, in_channels + dim, x_1, ..., x_dim)`
                with the coordinates in the last `dim` channels.
        """
        if data.ndim != self.dim + 2 or data.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected an input of shape (batch, {self.in_channels}, "
                f"{', '.join(['x'] * self.dim)}), got {tuple(data.shape)}."
            )
        batch_size, spatial_shape = data.shape[0], tuple(data.shape[2:])
        grids = [
            self.backend.math.broadcast_to(grid, (batch_size, 1, *spatial_shape))
            for grid in self._get_grids(data)
        ]
        return self.backend.math.concatenate([data, *grids], axis=1)

    def _get_grids(self, data: TensorType) -> list[TensorType]:
        """Returns the coordinates of the grid of the input, computing them only once.

        Args:
            data (TensorType): Input of shape `(batch, in_channels, x_1, ..., x_dim)`.

        Returns:
            list[TensorType]: For every dimension the coordinates, with the shape
                `(1, 1, x_1, ..., x_dim)`.
        """
        spatial_shape = tuple(data.shape[2:])
        boundaries = tuple(tuple(bounds) for bounds in self.grid_boundaries)
        key = (boundaries, spatial_shape, data.dtype, data.device)
        if key != self._grid_key:
            math = self.backend.math
            coordinates = []
            for resolution, (start, stop) in zip(spatial_shape, self.grid_boundaries):
                axis_coordinates = math.linspace(
                    start, stop, resolution + 1, dtype=data.dtype, device=data.device
                )
                coordinates.append(
                    axis_coordinates[:-1]
                )  # the right boundary is excluded
            self._grid = [
                math.expand_dims(math.expand_dims(grid, 0), 0)
                for grid in math.meshgrid(*coordinates, indexing="ij")
            ]
            self._grid_key = key
        return self._grid
