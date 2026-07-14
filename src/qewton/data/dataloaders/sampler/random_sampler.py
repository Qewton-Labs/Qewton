from qewton.data.dataloaders.sampler.point_sampler import PointSampler


class RandomUniformSampler(PointSampler):
    """Samples points uniformly at random from a geometry."""

    def sample_points(self):
        if self.filter_fn is None:
            return self._internal_sampling(self.batch_size)
        # with filtering
        points = self.backend.math.empty((0, self.geometry.dim), device=self._device)
        normals = self.backend.math.empty((0, self.geometry.dim), device=self._device)
        current_n = self.batch_size
        total_sampled_points = 0
        while len(points) < self.batch_size:
            total_sampled_points += current_n
            new_points, new_normals = self._internal_sampling(current_n)
            filter_fulfilled = self._evaluate_filter(new_points)
            new_points = new_points[filter_fulfilled]
            points = self.backend.math.concatenate((points, new_points), axis=0)
            if self.compute_normals:
                normals = self.backend.math.concatenate((normals, new_normals), axis=0)
            # Increase sampling number to get more points
            num_of_new_points = len(new_points)
            current_n = int(
                1.1
                * (self.batch_size - num_of_new_points)
                * total_sampled_points
                / num_of_new_points
            )
            current_n = max(min(1.0e6, current_n), 100)
        if self.compute_normals:
            return points[: self.batch_size], normals[: self.batch_size]
        return points[: self.batch_size], None

    def _internal_sampling(self, n_points):
        """Samples random uniform points from the geometry."""
        if self.is_boundary_geometry:
            sample_out = self.geometry.sample_random_uniform(
                n_points,
                device=self._device,
                include_normals=self.compute_normals,  # type: ignore
            )
            if self.compute_normals:
                return sample_out[0], sample_out[1]
            return sample_out, None
        return (
            self.geometry.sample_random_uniform(n_points, device=self._device),
            None,
        )
