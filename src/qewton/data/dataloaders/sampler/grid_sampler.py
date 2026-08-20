from typing import Callable

from qewton.backends import DEFAULT_DL_BACKEND, ComputingBackend, TensorType
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.base import Geometry
from qewton.graphs.nodes import NodeState
from qewton.optim.parameters.categorical_hyperparameter import CategoricalHyperparameter
from qewton.optim.parameters.number_hyperparameter import DiscreteHyperparameter

from qewton.data.dataloaders.sampler.point_sampler import PointSampler


class GridSampler(PointSampler[TensorType]):
    """Samples points in a grid-like fashion from a geometry.

    Args:
        geometry (Geometry): The geometry in which points should be sampled.
        n_points (int | Hyperparameter): The number of points that should be sampled.
        filter_fn (callable, optional): An additional filter_fn that specifies at which
            location points should be sampled. Internally a rejection sampling
            strategy is used. Default is None.
        compute_normals (bool, optional): Whether to compute normals to each sampled
            points. This is only possible for BoundaryGeometries. Defaults to False.
        normal_name (str | Variable): The name for the output port of the normals.
            Defaults to "normals".
        name (str, optional): The name of the node. Defaults to "PointSampler".
        state (NodeState, optional): The state of the node.
            Defaults to NodeState.FIXED.
    """

    def __init__(
        self,
        geometry: Geometry,
        n_points: int | DiscreteHyperparameter | CategoricalHyperparameter,
        filter_fn: Callable | None = None,
        compute_normals: bool = False,
        normal_name: str | Variable = "normals",
        name: str = "PointSampler",
        state: NodeState = NodeState.FIXED,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(
            geometry,
            n_points=n_points,
            filter_fn=filter_fn,
            compute_normals=compute_normals,
            normal_name=normal_name,
            name=name,
            state=state,
            backend=backend,
        )
        self.is_static = True
        self.point_cache: TensorType | None = None
        self.normal_cache: TensorType | None = None
        if self.filter_fn is not None:
            self.helper_sampler = RandomUniformSampler(
                geometry=self.geometry,
                n_points=1,
                filter_fn=self.filter_fn,
                compute_normals=self.compute_normals,
                backend=self.backend,
            )

    def sample_points(self):
        # Already created some points once
        if self.point_cache is not None:
            return self.point_cache, self.normal_cache

        if self.filter_fn is None:
            self.point_cache, self.normal_cache = self._direct_grid_sampling(
                self.batch_size
            )
            return self.point_cache, self.normal_cache
        self.point_cache, self.normal_cache = self._sample_with_filter()
        return self.point_cache, self.normal_cache

    def _sample_with_filter(self):
        # With filters sample first, then increase number of points:
        first_points, first_normals, correct_points, _ = (
            self._sample_grid_and_check_filter(self.batch_size)
        )
        # Found enough points
        if correct_points == self.batch_size:
            return first_points, first_normals
        # Else increase the number of points
        scaled_n = int(self.batch_size / (correct_points + 1)) * self.batch_size
        new_points, new_normals, correct_points, c_idx = (
            self._sample_grid_and_check_filter(scaled_n)
        )
        if correct_points >= self.batch_size:
            return self._slice_output(new_points, new_normals)
        # Finally add some random points
        self.helper_sampler._batch_size.current_value = (  # pylint: disable=W0212
            self.batch_size - correct_points
        )
        self.helper_sampler.to(self._device)
        # If we did not find any grid points just fall back to random
        if correct_points == 0:
            return self.helper_sampler.sample_points()
        rand_points, rand_normals = self.helper_sampler.sample_points()
        points = self.backend.math.concatenate(
            (new_points[c_idx], rand_points), axis=0  # type: ignore
        )
        normals = None
        if self.compute_normals:
            normals = self.backend.math.concatenate(
                (new_normals[c_idx], rand_normals), axis=0  # type: ignore
            )
        return points, normals

    def _slice_output(self, points, normals):
        if self.compute_normals:
            return points[: self.batch_size], normals[: self.batch_size]
        return points[: self.batch_size], None

    def _sample_grid_and_check_filter(self, n_points):
        first_points, first_normals = self._direct_grid_sampling(n_points)
        filter_fn_fulfilled = self._evaluate_filter(first_points)
        correct_points = self.backend.math.count_nonzero(filter_fn_fulfilled)
        return first_points, first_normals, correct_points, filter_fn_fulfilled

    def _direct_grid_sampling(self, n_points):
        """Samples points from the geometry in a grid-like fashion."""
        if self.has_boundary_geometry:
            sample_out = self.geometry.sample_grid(
                n_points,
                device=self._device,
                include_normals=self.compute_normals,  # type: ignore
            )
            if self.compute_normals:
                self.point_cache, self.normal_cache = sample_out[0], sample_out[1]
            else:
                self.point_cache, self.normal_cache = sample_out, None
        else:
            self.point_cache, self.normal_cache = (
                self.geometry.sample_grid(n_points, device=self._device),
                None,
            )
        return self.point_cache, self.normal_cache
