import numpy as np

from ..base import ContinuousGeometry, ContinuousBoundaryGeometry
from ....config.variables import Variable


class Triangle(ContinuousGeometry):
    """Class for triangles (NumPy-only, simplified).

    Parameters
    ----------
    variable : Variable
        The variable representing the underlying 2D space.
    origin, corner_1, corner_2 : array_like
        The three corners of the triangle in counter-clockwise order.
    """

    def __init__(
        self,
        variable: Variable,
        origin: np.ndarray | list[float] | tuple[float, float],
        corner_1: np.ndarray | list[float] | tuple[float, float],
        corner_2: np.ndarray | list[float] | tuple[float, float],
    ):
        assert variable.dim == 2
        self.origin = self._to_vector(origin)
        self.corner_1 = self._to_vector(corner_1)
        self.corner_2 = self._to_vector(corner_2)
        super().__init__(variable=variable)

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        points = points.reshape(-1, 2)
        origin = self.origin
        dir_1 = self.corner_1 - origin
        dir_2 = self.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.solve_barycentric(relative, dir_1, dir_2)
        inside = np.logical_and.reduce((bary_x >= 0, bary_y >= 0, bary_x + bary_y <= 1))
        return inside.reshape(-1, 1)

    def bounding_box(self):
        corners = np.vstack((self.origin, self.corner_1, self.corner_2))
        mins = np.min(corners, axis=0)
        maxs = np.max(corners, axis=0)
        return np.concatenate((mins[0:1], maxs[0:1], mins[1:2], maxs[1:2]))

    def sample_random_uniform(self, n_points: int):
        # use standard method to sample uniformly in triangle
        r1 = np.random.rand(n_points)
        r2 = np.random.rand(n_points)
        sqrt_r1 = np.sqrt(r1)
        b = sqrt_r1 * (1 - r2)
        c = sqrt_r1 * r2
        # point = a*A + b*B + c*C  -> convert to origin + b*(B-A) + c*(C-A)
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        points = self.origin + (b[:, None] * dir_1) + (c[:, None] * dir_2)
        return points

    def sample_grid(self, n_points: int):
        # build a barycentric grid and keep first n_points
        n_side = int(np.ceil(np.sqrt(2 * n_points)))
        u = np.linspace(0, 1, n_side)
        v = np.linspace(0, 1, n_side)
        uu, vv = np.meshgrid(u, v)
        bary = np.column_stack((uu.ravel(), vv.ravel()))
        mask = bary.sum(axis=1) <= 1
        bary = bary[mask]
        if len(bary) < n_points:
            # pad with random points if grid too coarse
            extra = n_points - len(bary)
            bary_rand = np.random.rand(extra, 2)
            sum_ge_1 = bary_rand.sum(axis=1) >= 1
            bary_rand[sum_ge_1] = 1 - bary_rand[sum_ge_1]
            bary = np.vstack((bary, bary_rand))
        random_choice = np.random.permutation(len(bary))[:n_points]
        bary = bary[random_choice]
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        points = self.origin + bary[:, :1] * dir_1 + bary[:, 1:] * dir_2
        return points

    def solve_barycentric(
        self, relative: np.ndarray, dir_1: np.ndarray, dir_2: np.ndarray
    ):
        # solve [dir_1 dir_2] [x; y] = relative.T for many points
        relative = np.asarray(relative, dtype=float).reshape(-1, 2)
        det = dir_1[0] * dir_2[1] - dir_1[1] * dir_2[0]
        if det == 0:
            raise ValueError("Degenerate triangle: direction vectors are collinear")
        bary_x = (dir_2[1] * relative[:, 0] - dir_2[0] * relative[:, 1]) / det
        bary_y = (-dir_1[1] * relative[:, 0] + dir_1[0] * relative[:, 1]) / det
        return bary_x, bary_y

    def _to_vector(self, vector):
        vec = np.asarray(vector, dtype=float)
        if vec.shape != (2,):
            raise ValueError("Triangle corner must be a 2D vector")
        return vec

    def _get_volume(self):
        # area of triangle
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        area = 0.5 * abs(dir_1[0] * dir_2[1] - dir_1[1] * dir_2[0])
        return area

    def create_boundary(self):
        return TriangleBoundary(self)


class TriangleBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry: Triangle):
        assert isinstance(geometry, Triangle)
        super().__init__(geometry)
        self.geometry: Triangle = geometry  # type: ignore

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        dir_1 = self.geometry.corner_1 - self.geometry.origin
        dir_2 = self.geometry.corner_2 - self.geometry.origin
        relative = points - self.geometry.origin
        bary_x, bary_y = self.geometry.solve_barycentric(relative, dir_1, dir_2)

        on_diagonal = np.isclose(bary_x + bary_y, 1)
        on_bottom = np.logical_and(np.isclose(bary_x, 0), (bary_y >= 0) & (bary_y <= 1))
        on_side = np.logical_and(np.isclose(bary_y, 0), (bary_x >= 0) & (bary_x <= 1))
        on_edge = np.logical_or.reduce((on_diagonal, on_bottom, on_side))
        return on_edge.reshape(-1, 1)

    def _get_volume(self):
        dir_1 = self.geometry.corner_1 - self.geometry.origin
        dir_2 = self.geometry.corner_2 - self.geometry.origin
        dir_3 = self.geometry.corner_1 - self.geometry.corner_2
        side_length = (
            np.linalg.norm(dir_1) + np.linalg.norm(dir_2) + np.linalg.norm(dir_3)
        )
        return side_length

    def sample_random_uniform(self, n_points: int):
        # sample uniformly along triangle edges proportional to edge length
        edges = [
            (self.geometry.origin, self.geometry.corner_1),
            (self.geometry.corner_1, self.geometry.corner_2),
            (self.geometry.corner_2, self.geometry.origin),
        ]
        lengths = np.array([np.linalg.norm(b - a) for a, b in edges])
        total = lengths.sum()
        probs = lengths / total
        choices = np.random.choice(3, size=n_points, p=probs)
        points = np.zeros((n_points, 2))
        for i in range(3):
            idx = np.where(choices == i)[0]
            if idx.size == 0:
                continue
            a, b = edges[i]
            t = np.random.rand(idx.size, 1)
            points[idx] = a + t * (b - a)
        return points

    def sample_grid(self, n_points: int):
        # distribute roughly equally across edges
        edges = [
            (self.geometry.origin, self.geometry.corner_1),
            (self.geometry.corner_1, self.geometry.corner_2),
            (self.geometry.corner_2, self.geometry.origin),
        ]
        lengths = np.array([np.linalg.norm(b - a) for a, b in edges])
        total = lengths.sum()
        counts = np.floor(n_points * (lengths / total)).astype(int)
        # ensure total count
        rem = n_points - counts.sum()
        for i in range(rem):
            counts[i % 3] += 1
        points = []
        for (a, b), c in zip(edges, counts):
            if c == 0:
                continue
            t = np.linspace(0, 1, c + 2)[1:-1][:, None] if c > 1 else np.array([[0.5]])
            pts = a + t * (b - a)
            points.append(pts)
        if len(points) == 0:
            return np.zeros((0, 2))
        points = np.vstack(points)
        if len(points) > n_points:
            points = points[:n_points]
        return points

    def normal(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        edges = [
            (self.geometry.origin, self.geometry.corner_1),
            (self.geometry.corner_1, self.geometry.corner_2),
            (self.geometry.corner_2, self.geometry.origin),
        ]
        mids = [0.5 * (a + b) for a, b in edges]
        edge_vecs = [b - a for a, b in edges]
        # normals (perp) and ensure outward by checking centroid
        centroid = (
            self.geometry.origin + self.geometry.corner_1 + self.geometry.corner_2
        ) / 3.0
        normals = []
        for vec, mid in zip(edge_vecs, mids):
            n = np.array([vec[1], -vec[0]])
            # if the normal points towards centroid, flip it to point outward
            if np.dot(n, centroid - mid) > 0:
                n = -n
            n = n / np.linalg.norm(n)
            normals.append(n)
        normals = np.array(normals)
        # compute barycentric coords to decide which edge each point belongs to
        dir_1 = self.geometry.corner_1 - self.geometry.origin
        dir_2 = self.geometry.corner_2 - self.geometry.origin
        bary_x, bary_y = self.geometry.solve_barycentric(
            points - self.geometry.origin, dir_1, dir_2
        )
        out = np.zeros_like(points)
        # edge mapping: x==0 -> edge CA (index 2), y==0 -> edge AB (index 0),
        # x+y==1 -> edge BC (index 1)
        idx_x0 = np.isclose(bary_x, 0)
        idx_y0 = np.isclose(bary_y, 0)
        idx_sum1 = np.isclose(bary_x + bary_y, 1)
        # Add normals (corners get sum of normals) then normalize
        out[idx_x0] += normals[2]
        out[idx_y0] += normals[0]
        out[idx_sum1] += normals[1]
        # handle corner cases where multiple conditions true
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
