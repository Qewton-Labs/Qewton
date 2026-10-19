from __future__ import annotations
from itertools import permutations

import numpy as np

from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.geometries.base import Geometry, DiscreteGeometry
from qewton.geometries.discrete.mesh import Mesh


def _as_numpy(data, backend) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(backend.to_numpy(data))


def factor_cells(factor) -> np.ndarray:
    """The cells of a factor's mesh as an integer array of shape
    (n_cells, simplex_dim + 1). A mesh without cells (e.g. the two end
    points of an interval boundary) contributes each vertex as a
    0-simplex."""
    cells = _as_numpy(factor.mesh.cells, factor.backend)
    if cells.ndim < 2 or cells.shape[0] == 0:
        return np.arange(len(factor.mesh.vertices)).reshape(-1, 1)
    return cells.astype(np.int64)


def _staircase_paths(simplex_dims: list[int]) -> np.ndarray:
    """Local vertex indices of every simplex in the staircase triangulation
    of a product of simplices with the given dimensions.

    Each simplex corresponds to a monotone lattice path from (0, ..., 0) to
    `simplex_dims`, advancing one factor per step.

    Returns:
        np.ndarray: Shape (n_paths, sum(simplex_dims) + 1, n_factors).
    """
    steps = [i for i, d in enumerate(simplex_dims) for _ in range(d)]
    paths = []
    for order in sorted(set(permutations(steps))):
        position = [0] * len(simplex_dims)
        path = [tuple(position)]
        for factor_idx in order:
            position[factor_idx] += 1
            path.append(tuple(position))
        paths.append(path)
    return np.asarray(paths, dtype=np.int64).reshape(
        -1, len(steps) + 1, len(simplex_dims)
    )


def product_cells(factors: list) -> np.ndarray:
    """Simplex cells of the Cartesian product of the factor meshes, with
    vertices indexed in product order (first factor slowest).

    Every product of factor cells is split by the staircase triangulation.
    Sorting each factor cell's vertices by their global index keeps the
    triangulation conforming across neighboring product cells.
    """
    cells = [np.sort(factor_cells(f), axis=1) for f in factors]
    sizes = [len(f.mesh.vertices) for f in factors]
    strides = np.cumprod([1] + sizes[::-1])[:-1][::-1]
    paths = _staircase_paths([c.shape[1] - 1 for c in cells])

    cell_combos = np.stack(
        np.meshgrid(*[np.arange(len(c)) for c in cells], indexing="ij"), axis=-1
    ).reshape(-1, len(cells))

    global_idx = np.zeros((len(cell_combos), *paths.shape[:2]), dtype=np.int64)
    for i, (factor_cell, stride) in enumerate(zip(cells, strides)):
        corners = factor_cell[cell_combos[:, i]]  # (n_combos, d_i + 1)
        global_idx += corners[:, paths[:, :, i]] * stride
    return global_idx.reshape(-1, paths.shape[1])


def product_points(factors: list) -> np.ndarray:
    """Vertices of the Cartesian product of the factor meshes in product
    order (first factor slowest), shape (prod(n_i), sum(dim_i))."""
    vertices = [_as_numpy(f.mesh.vertices, f.backend) for f in factors]
    sizes = [len(v) for v in vertices]
    columns = []
    for i, v in enumerate(vertices):
        shape = [1] * len(vertices) + [v.shape[-1]]
        shape[i] = sizes[i]
        columns.append(np.broadcast_to(v.reshape(shape), (*sizes, v.shape[-1])))
    return np.concatenate(columns, axis=-1).reshape(-1, sum(v.shape[-1] for v in vertices))


class ProductMeshGeometry(DiscreteGeometry[TensorType]):
    """The Cartesian product of meshed geometries, e.g. the discretization
    of a space-time domain.

    The vertices are all combinations of the factor vertices in product
    order, with the first factor varying slowest. Reshaping the vertex axis
    to `factor_shape` therefore gives one axis per factor.
    The simplex mesh of the full product is built on first access of
    `mesh`.

    Args:
        factors (list): The meshed factor geometries, each with a `mesh`
            and a `variable`, e.g. MeshGeometry objects.
        variable (Variable | None, optional): The variable of the product.
            Defaults to the composition of the factor variables.
        discretization_of (Geometry | None, optional): The geometry this
            mesh is a discretization of. Defaults to None.
        device (Device | str, optional): Where the vertices and cells are
            created. Defaults to cpu.
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        factors: list,
        variable: Variable | None = None,
        discretization_of: Geometry | None = None,
        device: Device | str = cpu,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        if len(factors) < 2:
            raise ValueError("A product mesh needs at least two factors.")
        if variable is None:
            variable = Variable.compose([f.variable for f in factors])
        points = product_points(factors)
        assert (
            points.shape[-1] == variable.dim
        ), "Dimension of the variable must match the summed factor dimensions."
        super().__init__(
            shape=(len(points),),
            variable=variable,
            dim=variable.dim,
            discretization_points=backend.build_tensor(points, device=device),
            backend=backend,
        )
        self.factors = list(factors)
        self.discretization_of = discretization_of
        self.device = device
        self._mesh: Mesh | None = None

    @property
    def factor_shape(self) -> tuple[int, ...]:
        """Number of vertices of each factor."""
        return tuple(len(f.mesh.vertices) for f in self.factors)

    @property
    def mesh(self) -> Mesh:
        """The simplex mesh of the full product.

        Raises:
            ValueError: If the product has more than 3 dimensions, which
                Mesh does not support.
        """
        if self._mesh is None:
            if self.dim > 3:
                raise ValueError(
                    f"The product of {[f.variable.name for f in self.factors]} is "
                    f"{self.dim}-dimensional, but meshes are supported up to 3D."
                )
            self._mesh = Mesh(
                vertices=self.discretization_points,
                cells=product_cells(self.factors),
                backend=self.backend,
                device=self.device,
            )
        return self._mesh

    def create_mesh(
        self, max_vertex_distance=None, device: Device = cpu
    ) -> ProductMeshGeometry:
        return self

    def bounding_box(self):
        points = _as_numpy(self.discretization_points, self.backend)
        bounds = np.stack([points.min(axis=0), points.max(axis=0)], axis=1)
        return self.backend.build_tensor(bounds.reshape(-1))


def _factor_points(factor) -> np.ndarray:
    """A factor's points with its own structural shape kept, e.g.
    (n_vertices, dim) for a mesh or (33, 33, dim) for a grid of points."""
    mesh = getattr(factor, "mesh", None)
    if mesh is not None:
        return _as_numpy(mesh.vertices, factor.backend)
    if factor.discretization_points is None:
        raise ValueError(
            f"Factor {factor.variable.name} has neither a mesh nor "
            "discretization points."
        )
    return _as_numpy(factor.discretization_points, factor.backend)


class ProductDiscreteGeometry(DiscreteGeometry[TensorType]):
    """The Cartesian product of discrete point sets, e.g. reference data
    known on a grid of space points for each of a few parameter values.

    Unlike ProductMeshGeometry, the factors need no mesh, and each factor
    keeps its own structural shape: factors of shapes (3,) and (33, 33)
    give a product of shape (3, 33, 33). The point coordinates are ordered
    as in `variable`, which may name the factors in a different order than
    `factors` - e.g. a sampler on `Omega * A_int` evaluated at reference
    data laid out angle first.

    Args:
        factors (list): Discrete geometries, each with a `variable` and
            either `discretization_points` or a `mesh`.
        variable (Variable | None, optional): The variable of the product,
            composed of the factor variables in any order. Defaults to their
            composition in the order of `factors`.
        backend (type[ComputingBackend[TensorType]] | None, optional):
            Defaults to the backend of the first factor.
    """

    def __init__(
        self,
        factors: list,
        variable: Variable | None = None,
        backend: type[ComputingBackend[TensorType]] | None = None,
    ):
        if len(factors) < 2:
            raise ValueError("A product needs at least two factors.")
        backend = backend or factors[0].backend
        if variable is None:
            variable = Variable.compose([f.variable for f in factors])
        if variable.dim != sum(f.variable.dim for f in factors):
            raise ValueError(
                f"{variable.name} has dim={variable.dim}, but the factors "
                f"{[f.variable.name for f in factors]} sum up to "
                f"{sum(f.variable.dim for f in factors)} dimensions."
            )

        factor_points = [_factor_points(f) for f in factors]
        factor_shapes = [p.shape[:-1] for p in factor_points]
        shape = tuple(n for s in factor_shapes for n in s)
        points = np.empty((*shape, variable.dim), dtype=factor_points[0].dtype)
        offset = 0
        for f, p, s in zip(factors, factor_points, factor_shapes):
            try:
                columns = variable.get_slice(f.variable)
            except KeyError as e:
                raise ValueError(
                    f"Factor {f.variable.name} is not part of {variable.name}."
                ) from e
            # Broadcast the factor's own axes into its place in the product.
            index = (None,) * offset + (slice(None),) * len(s)
            index += (None,) * (len(shape) - offset - len(s)) + (slice(None),)
            points[..., columns] = np.broadcast_to(
                p[index], (*shape, p.shape[-1])
            )
            offset += len(s)

        super().__init__(
            shape=shape,
            variable=variable,
            dim=variable.dim,
            discretization_points=backend.build_tensor(points),
            backend=backend,
        )
        self.factors = list(factors)

    @property
    def factor_shapes(self) -> list[tuple[int, ...]]:
        """The structural shape of each factor, in the order of `factors`."""
        return [tuple(_factor_points(f).shape[:-1]) for f in self.factors]
