from qewton.backends import Backend, DEFAULT_DL_BACKEND, TensorType
from qewton.config.axes import EllipsisAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.dtypes import Number
from qewton.graphs.nodes import GraphAwareNode, NodeState
from qewton.visualization.auto import auto_plot
from qewton.visualization.figure import Figure


class PlotNode(GraphAwareNode[TensorType]):
    """Graph-embedded sink: draws (and optionally saves) whatever
    array-like data reaches it, passing nothing downstream.

    `plot_type=None` (default) accepts any shape and picks a Plot via
    `auto_plot()`. Given a specific plot type, this is a plain pass-through
    to `auto_plot(x, config, plot_type=plot_type, **plot_kwargs)` -
    `plot_kwargs` must supply that type's required roles directly, exactly
    as constructing it directly would.

    The input port's own DataConfiguration is deliberately a wildcard
    (EllipsisAxes): auto_plot needs the *unified* DataConfiguration (which
    depends on whatever this node ends up connected to) to know what's
    actually plottable, and ports are typed once, before any connection
    exists - narrowing it upfront isn't possible, so an incompatible
    connection is only ever caught at forward() time, via auto_plot's/the
    underlying Plot's own validation.
    """

    # TODO: enhance this for dash apps etc later
    ellipsis_axes = EllipsisAxes()

    def __init__(
        self,
        plot_type: type | None = None,
        show: bool = True,
        save_path: str | None = None,
        name: str | None = None,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **plot_kwargs,
    ) -> None:
        self.plot_type = plot_type
        self.show = show
        self.save_path = save_path
        self.plot_kwargs = plot_kwargs
        self._graph = (
            None  # set by setup(), below - forward() has no other way to reach it
        )
        super().__init__(name=name, state=NodeState.FIXED, backend=backend)

    def setup(self, graph) -> None:
        self._graph = graph

    # No return annotation at all (not even `-> None`): Node._build_ports
    # only skips building an output port when the return annotation is
    # entirely absent - `-> None` itself still builds one real port typed
    # NoneType, which is not what a pure sink node wants.
    def forward(self, x: Number[TensorType, DataConfiguration(ellipsis_axes)]):
        port = self.input_ports[0]
        config = port.get_data_configuration(self._graph) or port.data_configuration
        plot = auto_plot(x, config, plot_type=self.plot_type, **self.plot_kwargs)
        figure = Figure(plot)
        if self.show:
            figure.show()
        if self.save_path is not None:
            figure.save_html(self.save_path)
