"""The renderer hierarchy: the default renderer is the first one whose
package is installed, Matplotlib the last."""

import subprocess
import sys

import pytest

from qewton.visualization.renderers import (
    DEFAULT_RENDERER,
    RENDERER_DICT,
    MatplotlibRenderer,
)

#: Blocks importing `packages` before importing qewton, then prints the
#: default renderer and the available ones.
_WITHOUT = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {packages}:
            raise ImportError(name)

sys.meta_path.insert(0, _Block())
import qewton
from qewton.visualization.renderers import DEFAULT_RENDERER, RENDERER_DICT
print(DEFAULT_RENDERER.__name__, *RENDERER_DICT)
"""

#: Missing packages: (default renderer, available renderers).
HIERARCHY = {
    "nothing": ("PlotlyRenderer", ["plotly", "matplotlib"]),
    "plotly": ("MatplotlibRenderer", ["matplotlib"]),
    "plotly and dash": ("MatplotlibRenderer", ["matplotlib"]),
}

#: Packages blocked per case.
BLOCKED = {"nothing": [], "plotly": ["plotly"], "plotly and dash": ["plotly", "dash"]}


@pytest.mark.parametrize("missing", list(HIERARCHY))
def test_the_default_renderer_is_the_first_available(missing):
    pytest.importorskip("plotly")
    script = _WITHOUT.format(packages=set(BLOCKED[missing]) or "set()")
    output = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    assert output == [HIERARCHY[missing][0], *HIERARCHY[missing][1]]


def test_matplotlib_is_always_available():
    assert list(RENDERER_DICT)[-1] == "matplotlib"
    assert RENDERER_DICT["matplotlib"] is MatplotlibRenderer
    assert DEFAULT_RENDERER is next(iter(RENDERER_DICT.values()))
