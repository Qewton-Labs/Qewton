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
