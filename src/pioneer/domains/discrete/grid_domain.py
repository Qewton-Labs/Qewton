from ..base import Domain


class GridDomain(Domain):

    def __init__(
        self,
        variable,
        bounding_box: list[tuple[float, float]],
        grid_size: list[int],
    ):
        super().__init__(variable, bounding_box)
        assert len(grid_size) == self.dim, "Grid size must match domain dimension."
        self.grid_size = grid_size
