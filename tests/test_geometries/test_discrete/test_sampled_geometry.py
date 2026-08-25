import copy

import numpy as np

from qewton.config.variables import Variable
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.sampled_geometry import SampledGeometry


def _source_geometry():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells = np.array([[0, 1, 2], [1, 3, 2]])
    return MeshGeometry(Variable("p", 2), Mesh(vertices=vertices, cells=cells))


class TestMeshProperty:
    def test_mesh_is_none_before_any_discretization(self):
        sampled = SampledGeometry(_source_geometry(), n_points=5)
        assert sampled.mesh is None

    def test_mesh_is_none_for_a_plain_point_cloud_without_cells(self):
        sampled = SampledGeometry(_source_geometry(), n_points=5)
        sampled.set_current_discretization(np.zeros((5, 2)), cells=None)
        assert sampled.mesh is None

    def test_mesh_is_populated_once_cells_are_set(self):
        sampled = SampledGeometry(_source_geometry(), n_points=4)
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cells = np.array([[0, 1, 2], [1, 3, 2]])
        sampled.set_current_discretization(vertices, cells)

        mesh = sampled.mesh
        assert mesh is not None
        assert np.asarray(mesh.vertices).shape == (4, 2)
        assert np.asarray(mesh.cells).shape == (2, 3)

    def test_discretization_points_reflects_the_current_state(self):
        sampled = SampledGeometry(_source_geometry(), n_points=3)
        assert sampled.discretization_points is None
        points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        sampled.set_current_discretization(points)
        assert np.allclose(np.asarray(sampled.discretization_points), points)


class TestToNumpy:
    def test_converts_points_and_cells_to_plain_numpy(self):
        import torch

        sampled = SampledGeometry(_source_geometry(), n_points=4)
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        sampled.set_current_discretization(vertices, cells)

        sampled.to_numpy()

        assert isinstance(sampled.discretization_points, np.ndarray)
        assert isinstance(sampled._current_cells, np.ndarray)
        assert np.allclose(sampled.discretization_points, vertices.numpy())

    def test_is_a_no_op_before_any_discretization(self):
        sampled = SampledGeometry(_source_geometry(), n_points=4)
        sampled.to_numpy()  # must not raise
        assert sampled.discretization_points is None

    def test_a_later_real_forward_overwrites_the_numpy_snapshot(self):
        """to_numpy() converts in place, but has no lasting effect on future
        sampling - the next set_current_discretization() call (a real
        forward()) replaces it with fresh backend tensors regardless."""
        import torch

        sampled = SampledGeometry(_source_geometry(), n_points=2)
        sampled.set_current_discretization(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        sampled.to_numpy()
        assert isinstance(sampled.discretization_points, np.ndarray)

        sampled.set_current_discretization(torch.tensor([[2.0, 2.0], [3.0, 3.0]]))
        assert isinstance(sampled.discretization_points, torch.Tensor)


class TestVisualizationMeshDeviceCaching:
    def test_different_device_arguments_get_separate_cache_entries(self):
        """Even though both resolve to the same actual device here (no GPU
        in this environment), device is part of the cache key - a call with
        an explicit device must never silently reuse a mesh built for a
        different (or unspecified) one."""
        sampled = SampledGeometry(_source_geometry(), n_points=4)
        sampled.visualization_mesh(None, None)
        sampled.visualization_mesh(None, "cpu")
        assert len(sampled._mesh_cache) == 2
        assert (None, None) in sampled._mesh_cache
        assert (None, "cpu") in sampled._mesh_cache

    def test_repeated_calls_with_the_same_key_reuse_the_cached_mesh(self):
        sampled = SampledGeometry(_source_geometry(), n_points=4)
        first = sampled.visualization_mesh(None, "cpu")
        second = sampled.visualization_mesh(None, "cpu")
        assert first is second


class TestDeepcopyIdentity:
    def test_deepcopy_returns_the_same_object(self):
        """SampledGeometry represents live, sampler-owned state (mutated by
        set_current_discretization() on every forward()) - a deepcopy must
        not silently disconnect a downstream DataConfiguration from it, so
        copy.deepcopy() is a no-op identity here, unlike every other
        Geometry."""
        sampled = SampledGeometry(_source_geometry(), n_points=3)
        assert copy.deepcopy(sampled) is sampled

    def test_deepcopy_inside_a_larger_structure_still_preserves_identity(self):
        sampled = SampledGeometry(_source_geometry(), n_points=3)
        copied_list = copy.deepcopy([sampled, {"geometry": sampled}])
        assert copied_list[0] is sampled
        assert copied_list[1]["geometry"] is sampled
