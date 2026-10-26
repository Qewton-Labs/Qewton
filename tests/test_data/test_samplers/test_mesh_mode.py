from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.point_sampler import discretization_mode
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle


def _sampler():
    X = Variable("x", 2)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    return GridSampler(square, 10)


class TestSetMeshModeDevice:
    def test_defaults_to_the_samplers_own_current_device(self):
        sampler = _sampler()
        sampler._device = "cpu"
        sampler.set_mesh_mode(max_vertex_distance=0.5)
        assert sampler.current_mesh_device == "cpu"

    def test_an_explicit_device_overrides_the_samplers_own(self):
        sampler = _sampler()
        sampler._device = "cpu"
        sampler.set_mesh_mode(max_vertex_distance=0.5, device="some-other-device")
        assert sampler.current_mesh_device == "some-other-device"

    def test_unset_mesh_mode_leaves_mesh_mode_false(self):
        sampler = _sampler()
        sampler.set_mesh_mode()
        sampler.unset_mesh_mode()
        assert sampler.mesh_mode is False


class TestDiscretizationModeContextManager:
    def test_passes_device_through_to_every_sampler(self):
        sampler_a, sampler_b = _sampler(), _sampler()
        with discretization_mode([sampler_a, sampler_b], None, device="cpu"):
            assert sampler_a.mesh_mode is True
            assert sampler_a.current_mesh_device == "cpu"
            assert sampler_b.current_mesh_device == "cpu"
        assert sampler_a.mesh_mode is False
        assert sampler_b.mesh_mode is False

    def test_no_device_argument_still_works(self):
        sampler = _sampler()
        with discretization_mode([sampler], None):
            assert sampler.mesh_mode is True
        assert sampler.mesh_mode is False


class TestVisualizationGeometry:
    def test_different_device_arguments_get_separate_cache_entries(self):
        """A call with an explicit device never reuses a mesh built for a
        different or unspecified one."""
        sampler = _sampler()
        sampler.visualization_geometry(None, None)
        sampler.visualization_geometry(None, "cpu")
        assert len(sampler._mesh_cache) == 2

    def test_repeated_calls_with_the_same_key_reuse_the_cached_mesh(self):
        sampler = _sampler()
        assert sampler.visualization_geometry(0.5, "cpu") is sampler.visualization_geometry(
            0.5, "cpu"
        )

    def test_a_copied_sampler_meshes_its_own_geometry(self):
        """In tuning, every trial deep-copies the trainer. The copy shares the
        SampledGeometry with the original, but meshes (and moves) only its
        own geometry."""
        import copy

        sampler = _sampler()
        copied = copy.deepcopy(sampler)
        assert copied.sampled_geometry is sampler.sampled_geometry
        assert copied.geometry is not sampler.geometry
        assert copied.visualization_geometry(0.5).discretization_of is copied.geometry
