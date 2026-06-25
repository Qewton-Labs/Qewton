from qewton.visualization.plot import Plot


class FeaturePlot(Plot):
    """Plot vector field data (2D or 3D quiver/arrow plots)."""

    def __init__(self, data, theme=None, title=None) -> None:
        super().__init__(data, theme, title)
        self.feature_dimension = ...
        self.geometry_dimension = ...
