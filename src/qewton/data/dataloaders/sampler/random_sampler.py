from qewton.data.dataloaders.sampler.point_sampler import PointSampler


class RandomUniformSampler(PointSampler):
    """Samples points uniformly at random from a geometry."""

    def sample_points(self):
        """Samples random uniform points from the geometry."""
        if self.is_boundary_geometry:
            return self.geometry.sample_random_uniform(
                self.batch_size,
                device=self._device,
                include_normals=self.compute_normals,  # type: ignore
            )
        return (
            self.geometry.sample_random_uniform(self.batch_size, device=self._device),
            None,
        )
