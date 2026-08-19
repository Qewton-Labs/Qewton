from plotly import graph_objects as go

from qewton.visualization.renderers.plotly.common import PlotlyArtist, _apply_scale


class ParallelCoordinatesArtist(PlotlyArtist):
    """One polyline per row, across one vertical axis per table column - the
    non-cartesian counterpart to ScatterArtist/BarArtist: go.Parcoords lays
    out its own axes internally, so this needs no x/y role at all, matching
    ParallelCoordinatesPlot.embedding_dim == None."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        line = dict()
        if result.color is not None:
            cmap = plot.color.cmap or plot.theme.default_cmap
            line.update(color=result.color, colorscale=cmap)
            line.update(_apply_scale(plot.color.scale))

        trace = go.Parcoords(dimensions=cls._dimensions(plot, result), line=line)
        backend_figure.add_trace(trace, row=row, col=col)
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)
        return cls(len(backend_figure.data) - 1)

    @staticmethod
    def _dimensions(plot, result):
        dimensions = []
        for key, column in result.columns.items():
            dim = dict(label=plot.labels.get(key, key), values=column.values)
            if column.labels is not None:
                dim.update(
                    tickvals=list(range(len(column.labels))), ticktext=column.labels
                )
            dimensions.append(dim)
        return dimensions

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        trace.dimensions = self._dimensions(plot, result)
        if result.color is not None:
            trace.line.color = result.color
            if plot.color.scale is not None:
                trace.line.cmin, trace.line.cmax = plot.color.scale.range

    def remove(self, backend_figure):
        pass
