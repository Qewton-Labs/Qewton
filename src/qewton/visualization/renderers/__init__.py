from .base import Renderer

#: The renderers whose packages are installed, in order of priority.
RENDERER_DICT: dict[str, type[Renderer]] = {}

try:
    from .plotly import PlotlyRenderer

    RENDERER_DICT["plotly"] = PlotlyRenderer
except ImportError:
    pass

# Matplotlib is a core dependency: the last renderer is always available.
from .matplotlib import MatplotlibRenderer  # noqa: E402

RENDERER_DICT["matplotlib"] = MatplotlibRenderer

#: Renderer class of a Figure created without a renderer: the first
#: available one.
DEFAULT_RENDERER = next(iter(RENDERER_DICT.values()))
