from dataclasses import dataclass

import numpy as np


@dataclass
class GridResult:
    """Result of a StructuredGridPlot family evaluate().

    x/y/z are reserved for real coordinate labels (see the "Tick coordinates"
    open item) - not populated yet, so artists still fall back to indices.
    """

    values: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    color: np.ndarray | None = None


@dataclass
class MeshResult:
    """Result of a MeshPlot family evaluate(), and of GeometryPlot."""

    vertices: np.ndarray
    cells: np.ndarray
    color: np.ndarray | None = None


@dataclass
class VectorResult:
    """Result of MeshVectorPlot.evaluate()."""

    positions: np.ndarray
    vectors: np.ndarray
    magnitude: np.ndarray


@dataclass
class CurveResult:
    """Result of LinePlot.evaluate() - one curve's x/y values.

    x falls back to plain indices until "Tick coordinates" lands, same as
    GridResult.x/y.
    """

    x: np.ndarray
    y: np.ndarray


@dataclass
class PathResult:
    """Result of PathPlot.evaluate() - an ordered sequence of positions in
    space (2D or 3D), e.g. a trajectory or streamline."""

    positions: np.ndarray
