import qewton
from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.spec import PlotSpec, SliderSpec, SelectorSpec
from enum import IntEnum
from dataclasses import dataclass

# TRAME
from trame.app import TrameApp
from trame.decorators import change
from trame.ui.vuetify3 import SinglePageLayout, SinglePageWithDrawerLayout
from trame.widgets import plotly, vtk, trame, client, vuetify3 as v3

# VTK
from vtkmodules.vtkCommonCore import vtkPoints, vtkLookupTable
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkCellLocator,
    vtkUnstructuredGrid,
    vtkPlane,
    vtkPolyData,
    VTK_TETRA,
    VTK_TRIANGLE,
)

from vtkmodules.vtkFiltersCore import (
    vtkCellCenters,
    vtkClipPolyData,
    vtkCutter,
    vtkPolyDataNormals,
    vtkTriangleFilter,
    vtkTubeFilter,
)
from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor, vtkScalarBarActor

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkDataSetMapper,
    vtkMapper,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

# Required for interactor initialization
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa

# Required for rendering initialization, not necessary for
# local rendering, but doesn't hurt to include it
import vtkmodules.vtkRenderingOpenGL2  # noqa

import numpy as np
from vtkmodules.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


Representation = IntEnum(
    "Representation", ["Points", "Wireframe", "Surface", "SurfaceWithEdges"]
)


class VectorComponent(IntEnum):
    """Component of a vector field used for coloring.

    The values match the component index expected by VTK, where ``-1`` selects
    the vector magnitude.
    """

    Magnitude = -1
    X = 0
    Y = 1
    Z = 2


VECTOR_COMPONENTS = [
    {"text": "Magnitude", "value": VectorComponent.Magnitude},
    {"text": "X", "value": VectorComponent.X},
    {"text": "Y", "value": VectorComponent.Y},
    {"text": "Z", "value": VectorComponent.Z},
]
COMPONENT_TEXT = {entry["value"]: entry["text"] for entry in VECTOR_COMPONENTS}

STREAMLINE_SEED_OFFSET_FACTOR = 1e-4  # relative to bounding box diagonal
STREAMLINE_MAX_SEEDS = 500
STREAMLINE_DEFAULT_SEEDS = 100

STREAMLINE_TUBE_RADIUS_FACTOR = 1.5e-3  # relative to the bounding box diagonal
STREAMLINE_TUBE_SIDES = 24

SLICE_MODES = [
    {"text": "Off", "value": "off"},
    {"text": "Slice", "value": "slice"},
    {"text": "Clip", "value": "clip"},
]

AXIS_NORMALS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "Custom": None,
}

# -----------------------------------------------------------------------------
# VTK STUFF
# -----------------------------------------------------------------------------


@dataclass
class VtkItem:
    name: str
    actor: vtkActor
    mapper: vtkMapper
    mesh: vtkPolyData | vtkUnstructuredGrid


@dataclass
class LUTData:
    name: str
    hue_range: tuple[float, float]
    saturation_range: tuple[float, float]
    value_range: tuple[float, float]


lookup_tables = [
    LUTData("Rainbow", (0.666, 0.0), (1.0, 1.0), (1.0, 1.0)),
    LUTData("Inv Rainbow", (0.0, 0.666), (1.0, 1.0), (1.0, 1.0)),
    LUTData("Greyscale", (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
    LUTData("Inv Greyscale", (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
    LUTData("Red", (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)),
    LUTData("Green", (0.333, 0.333), (1.0, 1.0), (1.0, 1.0)),
    LUTData("Blue", (0.666, 0.666), (1.0, 1.0), (1.0, 1.0)),
]


@dataclass
class ColorState:
    field: dict | None = None
    component: VectorComponent = VectorComponent.Magnitude
    preset: int = 0


class PolyDataBuffer:
    """Geometry shown by a mapper, decoupled from the producing filter.

    The mapper always refers to the same vtkPolyData instance, so the mapper
    input never has to be re-connected at runtime. In addition, an empty filter
    result is not forwarded: the previous geometry is kept and the caller can
    simply hide the actor until the filter produces cells again.
    """

    def __init__(self):
        self.data = vtkPolyData()

    def update_from(self, algorithm):
        """Pull `algorithm`; return True if it produced renderable geometry."""
        algorithm.Update()
        output = algorithm.GetOutput()
        if output is None or output.GetNumberOfCells() == 0:
            return False
        self.data.ShallowCopy(output)
        self.data.Modified()
        return True


class Streamlines:
    def __init__(self):
        self.enabled = False
        self.actor: vtkActor | None = None
        self.mapper: vtkMapper | None = None
        self.tracer: vtkStreamTracer | None = None
        self.tube: vtkTubeFilter | None = None
        self.tube_triangles: vtkTriangleFilter | None = None
        self.tube_clip: vtkClipPolyData | None = None
        self.buffer: PolyDataBuffer | None = None
        self.seed_polydata = None
        self.vector_field = None
        self.seed_surface_id: int | None = None
        self.n_seeds = STREAMLINE_DEFAULT_SEEDS
        self.n_available = 0

    def build(self, volume_mesh, plane: vtkPlane):
        tracer = vtkStreamTracer()
        tracer.SetInputData(volume_mesh)
        tracer.SetIntegratorTypeToRungeKutta45()
        tracer.SetIntegrationDirectionToBoth()
        tracer.SetMaximumPropagation(_bbox_diagonal(volume_mesh) * 10.0)
        tracer.SetInitialIntegrationStep(0.02)
        tracer.SetMaximumNumberOfSteps(5000)
        tracer.SetComputeVorticity(False)

        tube = vtkTubeFilter()
        tube.SetInputConnection(tracer.GetOutputPort())
        tube.SetRadius(_bbox_diagonal(volume_mesh) * STREAMLINE_TUBE_RADIUS_FACTOR)
        tube.SetNumberOfSides(STREAMLINE_TUBE_SIDES)
        tube.SetCapping(True)
        tube.SetVaryRadiusToVaryRadiusOff()

        # The tube filter emits triangle strips, which vtkClipPolyData cannot
        # handle. Always triangulate, so clipped and unclipped tubes come from
        # the exact same geometry.
        triangles = vtkTriangleFilter()
        triangles.SetInputConnection(tube.GetOutputPort())

        tube_clip = vtkClipPolyData()
        tube_clip.SetInputConnection(triangles.GetOutputPort())
        tube_clip.SetClipFunction(plane)

        buffer = PolyDataBuffer()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(buffer.data)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SetColorModeToMapScalars()
        mapper.InterpolateScalarsBeforeMappingOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2)  # only for tracer instead of tube
        actor.SetVisibility(False)

        self.tracer = tracer
        self.tube = tube
        self.tube_triangles = triangles
        self.tube_clip = tube_clip
        self.buffer = buffer
        self.mapper = mapper
        self.actor = actor

    def refresh(self, clip_enabled, invert):
        """Re-pull the tube geometry, optionally clipped. Returns True if shown."""
        if self.buffer is None or self.seed_polydata is None:
            return False
        self.tube_clip.SetInsideOut(invert)
        source = self.tube_clip if clip_enabled else self.tube_triangles
        return self.buffer.update_from(source)


class MeshClip:
    """Per-mesh clip filter that can be switched in/out of the mapper input."""

    def __init__(self, item: VtkItem, plane: vtkPlane):
        self.item = item
        if isinstance(item.mesh, vtkPolyData):
            clip = vtkClipPolyData()
        else:
            clip = vtkTableBasedClipDataSet()
        clip.SetInputData(item.mesh)
        clip.SetClipFunction(plane)
        self.clip = clip

    def set_enabled(self, enabled, invert=False):
        self.clip.SetInsideOut(invert)
        if enabled:
            self.item.mapper.SetInputConnection(self.clip.GetOutputPort())
        else:
            self.item.mapper.SetInputData(self.item.mesh)


class ClipSlice:
    """Single plane used for slicing or clipping the scene."""

    def __init__(self):
        self.plane = vtkPlane()
        self.cutter: vtkCutter | None = None
        self.buffer: PolyDataBuffer | None = None
        self.mapper: vtkPolyDataMapper | None = None
        self.actor: vtkActor | None = None
        self.mode = "off"
        self.invert = False
        self.bounds = (0, 1, 0, 1, 0, 1)
        self.center = (0.0, 0.0, 0.0)

    def set_bounds(self, volume_mesh):
        """Store the reference bounds and place the plane at the center."""
        self.bounds = volume_mesh.GetBounds()
        b = self.bounds
        self.center = (
            0.5 * (b[0] + b[1]),
            0.5 * (b[2] + b[3]),
            0.5 * (b[4] + b[5]),
        )
        self.plane.SetOrigin(*self.center)
        self.plane.SetNormal(0.0, 0.0, 1.0)

    def build(self, volume_mesh):
        """Create the slice pipeline once; returns the new actor (or None)."""
        if self.cutter is not None:
            return None

        cutter = vtkCutter()
        cutter.SetInputData(volume_mesh)
        cutter.SetCutFunction(self.plane)
        cutter.GenerateTrianglesOn()

        buffer = PolyDataBuffer()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(buffer.data)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SetColorModeToMapScalars()
        mapper.InterpolateScalarsBeforeMappingOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetAmbient(0.3)
        actor.SetVisibility(False)

        self.cutter, self.buffer, self.mapper, self.actor = cutter, buffer, mapper, actor
        return actor

    def refresh(self):
        """Re-pull the cut geometry. Returns True if the slice has geometry."""
        if self.buffer is None:
            return False
        return self.buffer.update_from(self.cutter)

    def set_plane(self, normal, offset):
        """`normal` in world coords, `offset` in [-1, 1] along the normal."""
        n = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            n, norm = np.array([0.0, 0.0, 1.0]), 1.0
        n /= norm

        b = self.bounds
        extent = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        # half-width of the bounding box projected onto the normal
        half = 0.5 * float(np.abs(n) @ extent)
        origin = np.asarray(self.center) + n * (offset * half)

        self.plane.SetNormal(*n)
        self.plane.SetOrigin(*origin)
        self.plane.Modified()


def create_points(vertices):
    """Create a VTK points object from the given vertices"""
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    return points


def create_mesh(vertices, cells, cell_type):
    """Create a VTK object from given vertices and cell connectivity"""
    vtk_points = create_points(vertices)
    cells = np.ascontiguousarray(cells, dtype=np.int64)

    n_vertices_per_cell = cells.shape[1]
    n_cells = cells.shape[0]

    # Cell connectivity
    offsets = np.arange(n_cells + 1, dtype=np.int64) * n_vertices_per_cell
    connectivity = cells.ravel()

    vtk_cells = vtkCellArray()
    vtk_cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )

    # Choose the appropriate VTK data structure
    if cell_type == VTK_TETRA:
        mesh = vtkUnstructuredGrid()
        mesh.SetPoints(vtk_points)
        mesh.SetCells(VTK_TETRA, vtk_cells)
    elif cell_type == VTK_TRIANGLE:
        mesh = vtkPolyData()
        mesh.SetPoints(vtk_points)
        mesh.SetPolys(vtk_cells)
    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")

    return mesh


def _bbox_diagonal(dataset):
    """Return the length of the bounding box diagonal of a VTK dataset."""
    b = dataset.GetBounds()
    return float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))


def _numpy_points_to_polydata(arr):
    """Wrap a numpy array of coordinates into a vtkPolyData point cloud."""
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(arr, dtype=np.float64)))
    pd = vtkPolyData()
    pd.SetPoints(points)
    return pd


def _count_inside(locator, points):
    """Count how many of the given points fall inside a cell of the located dataset."""
    return sum(1 for p in points if locator.FindCell(p.tolist()) >= 0)


def compute_seed_points(surface_mesh, volume_mesh, n_requested):
    """Build streamline seed points from the cells of a boundary surface.

    Seeds are placed at the cell centers of ``surface_mesh`` and pushed slightly
    into ``volume_mesh`` along the cell normals. The offset direction is
    determined geometrically by testing both candidate directions against the
    volume mesh, so the result does not depend on the normal orientation
    convention of the input file.

    Args:
        surface_mesh: vtkPolyData of the boundary patch used as seed source.
        volume_mesh: Volume mesh (e.g. tetrahedral) used for integration.
        n_requested: Desired number of seeds.

    Returns:
        Tuple ``(polydata, n_used, n_available)`` where ``polydata`` is the seed
        point cloud (or ``None`` if no valid seed could be created), ``n_used``
        the number of seeds actually contained in the returned polydata and
        ``n_available`` the number of cells on the boundary surface.
    """
    # Cell centers of the boundary patch
    centers = vtkCellCenters()
    centers.SetInputData(surface_mesh)
    centers.VertexCellsOn()
    centers.Update()
    pts = vtk_to_numpy(centers.GetOutput().GetPoints().GetData())

    n_available = len(pts)
    if n_available == 0:
        return None, 0, 0

    # Cell normals: orientation is intentionally left undetermined
    nrm_filter = vtkPolyDataNormals()
    nrm_filter.SetInputData(surface_mesh)
    nrm_filter.ComputePointNormalsOff()
    nrm_filter.ComputeCellNormalsOn()
    nrm_filter.SplittingOff()
    nrm_filter.ConsistencyOn()
    nrm_filter.AutoOrientNormalsOff()  # meaningless for open patches
    nrm_filter.Update()

    cell_normals = nrm_filter.GetOutput().GetCellData().GetNormals()
    nrm = vtk_to_numpy(cell_normals) if cell_normals is not None else None

    # Deterministic subsampling to the requested seed count
    n_target = min(int(n_requested), n_available)
    if n_available > n_target:
        idx = np.linspace(0, n_available - 1, n_target, dtype=int)
        pts = pts[idx]
        if nrm is not None:
            nrm = nrm[idx]

    # Cell locator on the volume mesh
    locator = vtkCellLocator()
    locator.SetDataSet(volume_mesh)
    locator.BuildLocator()

    if nrm is None:
        # No normals available: fall back to the unshifted cell centers.
        seeds = pts
    else:
        offset = _bbox_diagonal(volume_mesh) * STREAMLINE_SEED_OFFSET_FACTOR
        cand_neg = pts - nrm * offset  # against the normals
        cand_pos = pts + nrm * offset  # along the normals

        # Pick the direction that places more points inside the volume mesh.
        if _count_inside(locator, cand_neg) >= _count_inside(locator, cand_pos):
            seeds = cand_neg
        else:
            seeds = cand_pos

    # Drop seeds that ended up outside the volume mesh
    valid = np.array([locator.FindCell(p.tolist()) >= 0 for p in seeds], dtype=bool)
    seeds = seeds[valid]

    n_used = len(seeds)
    if n_used == 0:
        return None, 0, n_available

    return _numpy_points_to_polydata(seeds), n_used, n_available


def init_renderer():
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
    return renderer, renderWindow


# -----------------------------------------------------------------------------
# Callbacks for VTK actors
# -----------------------------------------------------------------------------


# Representation Callback
def update_representation(actor, mode):
    property = actor.GetProperty()
    if mode == Representation.Points:
        property.SetRepresentationToPoints()
        property.SetPointSize(5)
        property.EdgeVisibilityOff()
    elif mode == Representation.Wireframe:
        property.SetRepresentationToWireframe()
        property.SetPointSize(1)
        property.EdgeVisibilityOff()
    elif mode == Representation.Surface:
        property.SetRepresentationToSurface()
        property.SetPointSize(1)
        property.EdgeVisibilityOff()
    elif mode == Representation.SurfaceWithEdges:
        property.SetRepresentationToSurface()
        property.SetPointSize(1)
        property.EdgeVisibilityOn()


def apply_lut_preset(lut, preset):
    """Configure a lookup table from the preset table."""
    if lut is None:
        return
    table = lookup_tables[int(preset)]
    lut.SetHueRange(*table.hue_range)
    lut.SetSaturationRange(*table.saturation_range)
    lut.SetValueRange(*table.value_range)
    lut.Build()


def apply_color_state(mapper, state: ColorState, lut):
    """Configure `mapper` to color by `state` using `lut`."""
    if mapper is None:
        return

    field = state.field
    if field is None or lut is None:
        mapper.ScalarVisibilityOff()
        return

    name = field["text"]
    mapper.SetLookupTable(lut)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(name)
    mapper.SetColorModeToMapScalars()
    mapper.InterpolateScalarsBeforeMappingOn()

    if field["type"] == "scalar":
        mapper.SetScalarRange(*field["range"])
        lut.SetVectorModeToComponent()
        lut.SetVectorComponent(0)
    else:
        comp = state.component
        mapper.SetScalarRange(*field["component_ranges"][comp])
        if comp == VectorComponent.Magnitude:
            lut.SetVectorModeToMagnitude()
        else:
            lut.SetVectorModeToComponent()
            lut.SetVectorComponent(int(comp))
        mapper.ColorByArrayComponent(name, int(comp))

    mapper.ScalarVisibilityOn()


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


class MeshScene:
    def __init__(self) -> None:
        self.actors = []

    def build(
        self, qmesh: "qewton.geometries.MeshGeometry", scalar_fields, vector_fields
    ):
        self._init_cube_axes()
        tet_mesh = create_mesh(qmesh.mesh.vertices, qmesh.mesh.cells, VTK_TETRA)
        self._init_scalar_fields(tet_mesh, scalar_fields)
        self._init_vector_fields(tet_mesh, vector_fields)
        self.vtk_color_fields = self.vtk_scalar_fields + self.vtk_vector_fields
        self._init_meshes(qmesh, tet_mesh)
        self._init_colors()
        self._init_clip_slice()
        self._init_streamlines()
        self._init_scalar_bar()

    def _init_cube_axes(self):
        self.cubeAxes = vtkCubeAxesActor()
        self.actors.append(self.cubeAxes)

        # Cube Axes: camera, and styling
        self.cubeAxes.SetXLabelFormat("%6.1f")
        self.cubeAxes.SetYLabelFormat("%6.1f")
        self.cubeAxes.SetZLabelFormat("%6.1f")
        self.cubeAxes.SetFlyModeToOuterEdges()

    def _init_scalar_bar(self):
        mapper = self.volume_mesh.mapper
        bar = vtkScalarBarActor()
        bar.SetLookupTable(mapper.GetLookupTable())
        bar.SetNumberOfLabels(5)
        bar.SetOrientationToVertical()
        bar.SetWidth(0.08)
        bar.SetHeight(0.45)
        bar.SetPosition(0.88, 0.5)
        bar.SetLabelFormat("%.3f")
        bar.SetTitle("")
        bar.GetTitleTextProperty().SetJustificationToLeft()
        bar.GetLabelTextProperty().SetJustificationToLeft()
        self.scalar_bar = bar
        self.actors.append(bar)

    def _init_meshes(self, qmesh: "qewton.geometries.MeshGeometry", tet_mesh):
        tet_mapper = vtkDataSetMapper()
        tet_mapper.SetInputData(tet_mesh)
        tet_actor = vtkActor()
        tet_actor.SetMapper(tet_mapper)
        self.actors.append(tet_actor)
        self.cubeAxes.SetBounds(tet_actor.GetBounds())

        self.volume_mesh = VtkItem("Mesh", tet_actor, tet_mapper, tet_mesh)
        self.vtk_meshes = [self.volume_mesh]

        submeshes = [
            name for name, value in qmesh.mesh.marker_labels.items() if value[1] == 2
        ]
        for i, surf_name in enumerate(submeshes):
            surface_geo = qmesh.boundary.get_submesh(surf_name)
            surface_mesh = create_mesh(
                surface_geo.mesh.vertices, surface_geo.mesh.cells, VTK_TRIANGLE
            )
            surface_mapper = vtkDataSetMapper()
            surface_mapper.SetInputData(surface_mesh)
            surface_actor = vtkActor()
            surface_actor.SetMapper(surface_mapper)
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            surface_actor.GetProperty().SetColor(*colors[i % 3])
            self.actors.append(surface_actor)
            self.vtk_meshes.append(
                VtkItem(surf_name, surface_actor, surface_mapper, surface_mesh)
            )

    def _init_scalar_fields(self, mesh, fields: dict):
        self.vtk_scalar_fields = []
        for name, f in fields.items():
            values = np.asarray(f, dtype=np.float64)
            vtk_values = numpy_to_vtk(values, deep=True, array_type=None)
            vtk_values.SetName(name)
            mesh.GetPointData().AddArray(vtk_values)
            self.vtk_scalar_fields.append(
                {
                    "text": name,
                    "value": name,
                    "type": "scalar",
                    "range": [values.min(), values.max()],
                }
            )

    def _init_vector_fields(self, mesh, vector_fields: dict):
        self.vtk_vector_fields = []
        for name, v in vector_fields.items():
            values = np.asarray(v, dtype=np.float64)
            assert values.ndim == 2 and (values.shape[1] == 2 or values.shape[1] == 3)

            vtk_values = numpy_to_vtk(values, deep=True, array_type=None)
            dim = values.shape[1]
            vtk_values.SetNumberOfComponents(dim)
            vtk_values.SetName(name)
            mesh.GetPointData().AddArray(vtk_values)

            magnitudes = np.linalg.norm(values, axis=1)
            component_ranges = {
                VectorComponent.X: [values[:, 0].min(), values[:, 0].max()],
                VectorComponent.Y: [values[:, 1].min(), values[:, 1].max()],
                VectorComponent.Z: (
                    [values[:, 2].min(), values[:, 2].max()] if dim == 3 else None
                ),
                VectorComponent.Magnitude: [magnitudes.min(), magnitudes.max()],
            }

            self.vtk_vector_fields.append(
                {
                    "text": name,
                    "value": name,
                    "type": "vector",
                    "component_ranges": component_ranges,
                }
            )

    # -------------------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------------------

    def _init_colors(self):
        """One authoritative color state per mesh; mesh 0 drives derived actors."""
        self.color_states = [ColorState() for _ in self.vtk_meshes]
        for mesh_id, state in enumerate(self.color_states):
            apply_lut_preset(
                self.vtk_meshes[mesh_id].mapper.GetLookupTable(), state.preset
            )

    def set_color_field(self, mesh_id, field, component=None):
        """Select the colored field (and vector component) of a mesh."""
        state = self.color_states[mesh_id]
        state.field = field
        state.component = (
            VectorComponent(component)
            if component is not None
            else VectorComponent.Magnitude
        )
        self._apply_colors(mesh_id)

    def set_color_preset(self, mesh_id, preset):
        """Select the colormap preset of a mesh."""
        state = self.color_states[mesh_id]
        state.preset = int(preset)
        apply_lut_preset(self.vtk_meshes[mesh_id].mapper.GetLookupTable(), state.preset)
        self._apply_colors(mesh_id)

    def _apply_colors(self, mesh_id):
        """Push the stored color state of a mesh into VTK and sync dependents."""
        item = self.vtk_meshes[mesh_id]
        state = self.color_states[mesh_id]
        apply_color_state(item.mapper, state, item.mapper.GetLookupTable())
        if mesh_id == 0:
            self.sync_derived_colors()
            self.update_scalar_bar()

    def sync_derived_colors(self):
        """Re-apply the volume mesh color state to slice + streamlines."""
        state = self.color_states[0]
        lut = self.volume_mesh.mapper.GetLookupTable()
        apply_color_state(self.vtk_clip_slice.mapper, state, lut)
        apply_color_state(self.vtk_streamlines.mapper, state, lut)

    def update_scalar_bar(self):
        """Sync scalar bar title and LUT with the volume mesh color state."""
        state = self.color_states[0]
        field = state.field
        if field is None:
            return

        if field["type"] == "scalar":
            title = field["text"]
        else:
            title = f"{field['text']} ({COMPONENT_TEXT[state.component]})"

        bar = self.scalar_bar
        bar.SetTitle(title)
        bar.SetLookupTable(self.volume_mesh.mapper.GetLookupTable())
        bar.Modified()

    # -------------------------------------------------------------------------
    # Streamlines
    # -------------------------------------------------------------------------

    def _init_streamlines(self):
        self.vtk_streamlines = Streamlines()

        # Boundary surfaces for seed selector
        self.vtk_seed_surfaces = [
            {"text": mesh.name, "value": idx}
            for idx, mesh in enumerate(self.vtk_meshes)
            if idx != 0  # Don't include volume mesh itself
        ]

    def _build_streamlines(self):
        """Create Tracer/Mapper/Actor once upon first use."""
        sl = self.vtk_streamlines
        if sl.tracer is not None:
            return None

        sl.build(self.volume_mesh.mesh, self.vtk_clip_slice.plane)
        assert sl.actor
        self.actors.append(sl.actor)
        return sl.actor

    def update_streamlines(self, streamlines_enabled, field, surface_id, n_requested):
        sl = self.vtk_streamlines
        cs = self.vtk_clip_slice
        sl.enabled = bool(streamlines_enabled)
        hint = None
        new_actor = None

        if not sl.enabled:
            if sl.actor is not None:
                sl.actor.SetVisibility(False)
            return (hint, new_actor)

        new_actor = self._build_streamlines()

        if field is None or surface_id is None:
            sl.actor.SetVisibility(False)
            return (hint, new_actor)

        # Set active vector field for volume mesh
        volume_mesh = self.volume_mesh.mesh
        volume_mesh.GetPointData().SetActiveVectors(field)

        # Compute seeds
        surface_mesh = self.vtk_meshes[surface_id].mesh
        seed_pd, n_used, n_available = compute_seed_points(
            surface_mesh, volume_mesh, n_requested
        )

        sl.seed_polydata = seed_pd
        sl.vector_field = field
        sl.seed_surface_id = surface_id
        sl.n_seeds = n_used
        sl.n_available = n_available

        if seed_pd is None:
            sl.actor.SetVisibility(False)
            hint = "No seeds available"
            return (hint, new_actor)

        hint = f"{n_used} seeds from {n_available} faces"

        sl.tracer.SetSourceData(seed_pd)

        self.sync_derived_colors()

        sl.actor.SetVisibility(sl.refresh(cs.mode == "clip", cs.invert))
        return (hint, new_actor)

    # -------------------------------------------------------------------------
    # Slicing / clipping
    # -------------------------------------------------------------------------

    def _init_clip_slice(self):
        self.vtk_clip_slice = ClipSlice()
        self.vtk_clip_slice.set_bounds(self.volume_mesh.mesh)
        self.mesh_clips = [
            MeshClip(item, self.vtk_clip_slice.plane) for item in self.vtk_meshes
        ]

    def update_clip_slice(self, mode, normal, offset, invert):
        """Apply slice/clip state. Returns a newly created actor or None."""
        cs = self.vtk_clip_slice
        cs.mode = mode
        # Inverting is only meaningful for clipping, where it selects the kept
        # side without moving the plane.
        cs.invert = bool(invert) and mode == "clip"
        new_actor = None

        if mode != "off":
            new_actor = cs.build(self.volume_mesh.mesh)

        cs.set_plane(normal, offset)

        # clipping: real geometry, so it also works with local rendering
        clip_on = mode == "clip"
        for mesh_clip in self.mesh_clips:
            mesh_clip.set_enabled(clip_on, cs.invert)

        sl = self.vtk_streamlines
        if sl.actor is not None:
            sl.actor.SetVisibility(sl.enabled and sl.refresh(clip_on, cs.invert))

        # slicing
        if cs.actor is not None:
            if mode == "slice":
                self.sync_derived_colors()
                cs.actor.SetVisibility(cs.refresh())
            else:
                cs.actor.SetVisibility(False)

        return new_actor


# -----------------------------------------------------------------------------
# GUI elements
# -----------------------------------------------------------------------------


class TrameApplication(TrameApp, RenderApplication):
    """Serves a Figure as a Trame app, supporting Plotly or VTK based backends"""

    def __init__(self, server=None):
        super().__init__(server, client_type="vue3")
        self.title = None
        self.state.setdefault("active_ui", None)

    @staticmethod
    def create(figure):
        """Builds a Trame app displaying `figure`"""
        # TODO: implement sliders / callbacks
        app = TrameApplication()

        app._build_figure_ui(figure)
        # _build_ui._register_callbacks(figure)
        return app

    @staticmethod
    def createMesh(mesh, scalar_fields, vector_fields):
        """Builds a Trame app displaying `mesh`, colored by `scalar_fields` or `vector_fields`"""
        app = TrameApplication()

        app._build_mesh_ui(mesh, scalar_fields, vector_fields)
        return app

    def _hide_kitware_link(self):
        # Inject CSS to hide ONLY the Kitware website link inside the footer
        client.Style(".v-footer a[href*='kitware.com'] { display: none !important; }")

    def _build_figure_ui(self, figure):
        # with SinglePageWithDrawerLayout(self.server, theme=("theme", "light"), hide_footer=True) as self.ui:
        with SinglePageLayout(self.server, full_height=True) as self.ui:
            self._hide_kitware_link()
            title = figure.title if figure.title else "Plot"
            self.ui.title.set_text("Qewton Viewer: " + title)
            self.state.trame__title = "Qewton"
            self.ui.icon.hide()

            with self.ui.content:
                pfig = plotly.Figure(
                    ctx_name="plotly_display",
                    display_logo=False,
                    display_mode_bar=True,
                )
                pfig.update(figure.draw())

    def _build_mesh_ui(self, mesh, scalar_fields, vector_fields):
        self._init_vtk()
        self._init_mesh(mesh, scalar_fields, vector_fields)
        sc = self.scene

        with SinglePageWithDrawerLayout(
            self.server, theme=("theme", "light"), hide_footer=True
        ) as self.ui:
            self.state.vtk_field_types = {
                f["value"]: f["type"] for f in sc.vtk_color_fields
            }

            self._hide_kitware_link()
            title = "Mesh"  # mesh.title if mesh.title else "Mesh"
            self.ui.title.set_text("Qewton VTK Viewer: " + title)
            self.state.trame__title = "Qewton"

            with self.ui.toolbar:
                # toolbar components
                v3.VSpacer()
                v3.VDivider(vertical=True, classes="mx-2")
                self.standard_buttons()

            with self.ui.footer as footer:
                v3.VLabel("Qewton Viewer")

            with self.ui.content:
                # content components
                with v3.VContainer(
                    fluid=True,
                    classes="pa-0 fill-height",
                ):
                    # view = vtk.VtkRemoteView(self.renderWindow, interactive_ratio=1)
                    view = vtk.VtkLocalView(self.renderWindow)
                    # view = vtk.VtkRemoteLocalView(
                    #    self.renderWindow, namespace="view", mode="local", interactive_ratio=1
                    # )
                    self.ctrl.view_update = view.update
                    self.ctrl.view_reset_camera = view.reset_camera

            with self.ui.drawer as drawer:
                # drawer components
                drawer.width = 350
                self.pipeline_widget()
                v3.VDivider(classes="mb-2")
                for i, m in enumerate(sc.vtk_meshes):
                    vol_mesh = i == 0
                    self.mesh_card(
                        m.name,
                        i,
                        show_color=vol_mesh,
                        show_streamlines=vol_mesh,
                        show_slice=vol_mesh,
                    )

    def _init_vtk(self):
        self.renderer, self.renderWindow = init_renderer()

    def _init_streamlines_state(self):
        sc = self.scene
        self.state.streamlines_enabled = False
        self.state.streamlines_field = (
            sc.vtk_vector_fields[0]["value"] if sc.vtk_vector_fields else None
        )
        self.state.streamlines_seed_surface = (
            sc.vtk_seed_surfaces[0]["value"] if sc.vtk_seed_surfaces else None
        )
        self.state.streamlines_n_seeds = STREAMLINE_DEFAULT_SEEDS
        self.state.streamlines_seed_hint = ""

    def _init_clip_slice_state(self):
        self.state.slice_mode = "off"
        self.state.slice_axis = "Z"
        self.state.slice_offset = 0.0
        self.state.slice_invert = False
        # Separate scalars: trame does not track in-place mutation of list items.
        self.state.slice_nx = 0.0
        self.state.slice_ny = 0.0
        self.state.slice_nz = 1.0

    def _init_mesh(
        self, qmesh: "qewton.geometries.MeshGeometry", scalar_fields, vector_fields
    ):
        self.scene = MeshScene()
        self.scene.build(qmesh, scalar_fields, vector_fields)
        self.scene.cubeAxes.SetCamera(self.renderer.GetActiveCamera())
        self._init_streamlines_state()
        self._init_clip_slice_state()
        for actor in self.scene.actors:
            self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

    # -----------------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------------

    @change("cube_axes_visibility")
    def update_cube_axes_visibility(self, cube_axes_visibility, **_):
        self.scene.cubeAxes.SetVisibility(cube_axes_visibility)
        self.ctrl.view_update()

    @change("colorbar_visible")
    def update_colorbar_visibility(self, colorbar_visible, **_):
        self.scene.scalar_bar.SetVisibility(colorbar_visible)
        self.ctrl.view_update()

    # Selection Change
    def actives_change(self, ids):
        id = ids[0]
        self.state.active_ui = f"card_id_{id}"

    # Visibility Change
    def visibility_change(self, event):
        id = event["id"]
        idx = int(id) - 1
        visibility = event["visible"]

        self.scene.vtk_meshes[idx].actor.SetVisibility(visibility)
        self.ctrl.view_update()

    def update_mesh_representation(self, mesh_id, mesh_representation, **_):
        mesh_actor = self.scene.vtk_meshes[mesh_id].actor
        update_representation(mesh_actor, mesh_representation)
        self.ctrl.view_update()

    def update_mesh_opacity(self, mesh_id, mesh_opacity, **_):
        mesh_actor = self.scene.vtk_meshes[mesh_id].actor
        mesh_actor.GetProperty().SetOpacity(mesh_opacity)
        self.ctrl.view_update()

    def update_mesh_color_preset(self, mesh_id, mesh_color_preset, **_):
        self.scene.set_color_preset(mesh_id, mesh_color_preset)
        self.ctrl.view_update()

    def update_mesh_color_array(
        self, mesh_id, mesh_color_field, mesh_color_component=None, **_
    ):
        self.scene.set_color_field(mesh_id, mesh_color_field, mesh_color_component)
        self.ctrl.view_update()

    @change(
        "streamlines_enabled",
        "streamlines_field",
        "streamlines_seed_surface",
        "streamlines_n_seeds",
    )
    def update_streamlines(self, **_):
        enabled = self.state.streamlines_enabled
        field = self.state.streamlines_field
        surface_id = self.state.streamlines_seed_surface
        n_requested = self.state.streamlines_n_seeds
        hint, new_actor = self.scene.update_streamlines(
            enabled, field, surface_id, n_requested
        )
        if hint:
            self.state.streamlines_seed_hint = hint
        if new_actor:
            self.renderer.AddActor(new_actor)
        self.ctrl.view_update()

    @change(
        "slice_mode",
        "slice_axis",
        "slice_nx",
        "slice_ny",
        "slice_nz",
        "slice_offset",
        "slice_invert",
    )
    def update_clip_slice(self, **_):
        axis = self.state.slice_axis
        preset = AXIS_NORMALS.get(axis)
        if preset is not None:
            normal = preset
            # Keep the custom fields in sync, so switching to "Custom"
            # starts from the currently shown plane.
            self.state.slice_nx, self.state.slice_ny, self.state.slice_nz = preset
        else:
            normal = (
                float(self.state.slice_nx or 0.0),
                float(self.state.slice_ny or 0.0),
                float(self.state.slice_nz or 0.0),
            )
        new_actor = self.scene.update_clip_slice(
            self.state.slice_mode,
            normal,
            float(self.state.slice_offset),
            bool(self.state.slice_invert),
        )
        if new_actor is not None:
            self.renderer.AddActor(new_actor)
        self.ctrl.view_update()

    def _make_representation_callback(self, mesh_id):
        def on_change(**kwargs):
            ui_id = str(mesh_id + 1)
            value = kwargs[f"mesh_representation_{ui_id}"]
            self.update_mesh_representation(mesh_id, value)

        return on_change

    def _make_opacity_callback(self, mesh_id):
        def on_change(**kwargs):
            ui_id = str(mesh_id + 1)
            value = kwargs[f"mesh_opacity_{ui_id}"]
            self.update_mesh_opacity(mesh_id, value)

        return on_change

    def _make_color_preset_callback(self, mesh_id):
        def on_change(**kwargs):
            ui_id = str(mesh_id + 1)
            value = kwargs[f"mesh_color_preset_{ui_id}"]
            self.update_mesh_color_preset(mesh_id, value)

        return on_change

    def _make_color_field_callback(self, mesh_id):
        def on_change(**kwargs):
            ui_id = str(mesh_id + 1)
            field_name = kwargs[f"mesh_color_field_{ui_id}"]
            component = kwargs.get(f"mesh_color_component_{ui_id}", None)
            field = next(
                f for f in self.scene.vtk_color_fields if f["value"] == field_name
            )
            # Explicitly set state variable to control visibility
            setattr(
                self.state, f"mesh_color_is_vector_{ui_id}", field["type"] == "vector"
            )
            self.update_mesh_color_array(mesh_id, field, component)

        return on_change

    # -----------------------------------------------------------------------------
    # GUI elements
    # -----------------------------------------------------------------------------

    def standard_buttons(self):
        v3.VCheckbox(
            v_model=("cube_axes_visibility", True),
            true_icon="mdi-cube-outline",
            false_icon="mdi-cube-off-outline",
            classes="mx-1",
            hide_details=True,
            density="compact",
        )
        v3.VCheckbox(
            v_model=("colorbar_visible", True),
            true_icon="mdi-palette",
            false_icon="mdi-palette-outline",
            classes="mx-1",
            hide_details=True,
            density="compact",
        )
        v3.VCheckbox(
            v_model="theme",
            true_value="dark",
            false_value="light",
            true_icon="mdi-lightbulb-off-outline",
            false_icon="mdi-lightbulb-outline",
            classes="mx-1",
            hide_details=True,
            density="compact",
        )
        v3.VCheckbox(
            v_model=("viewMode", "local"),
            true_icon="mdi-lan-disconnect",
            false_icon="mdi-lan-connect",
            true_value="local",
            false_value="remote",
            classes="mx-1",
            hide_details=True,
            density="compact",
        )
        v3.VBtn(icon="mdi-crop-free", click=self.ctrl.view_reset_camera)

    def pipeline_widget(self):
        assert len(self.scene.vtk_meshes) >= 1
        pipeline = []
        for i, m in enumerate(self.scene.vtk_meshes):
            # p = {"id": str(i+1), "parent": "0" if i == 0 else "1", "visible": 1, "name": m.name}
            p = {"id": str(i + 1), "parent": str(i), "visible": 1, "name": m.name}
            pipeline.append(p)
        trame.GitTree(
            sources=("pipeline", pipeline),
            actives_change=(self.actives_change, "[$event]"),
            visibility_change=(self.visibility_change, "[$event]"),
        )

    def ui_card(self, title, ui_id):
        with v3.VCard(v_show=f"active_ui == 'card_id_{ui_id}'"):
            v3.VCardTitle(
                title,
                classes="grey lighten-1 py-1 grey--text text--darken-3",
                style="user-select: none; cursor: pointer",
                hide_details=True,
                density="compact",
            )
            content = v3.VCardText(classes="py-2")
        return content

    def mesh_card(self, title, mesh_id, show_color, show_streamlines, show_slice):
        ui_id = str(mesh_id + 1)
        sc = self.scene
        default_field = self.scene.vtk_color_fields[0]
        self.state[f"mesh_color_is_vector_{ui_id}"] = default_field["type"] == "vector"
        with self.ui_card(title=title, ui_id=ui_id):
            # Representation
            v3.VSelect(
                v_model=(f"mesh_representation_{ui_id}", Representation.Surface),
                items=(
                    "representations",
                    [
                        {"text": "Points", "value": Representation.Points},
                        {"text": "Wireframe", "value": Representation.Wireframe},
                        {"text": "Surface", "value": Representation.Surface},
                        {
                            "text": "SurfaceWithEdges",
                            "value": Representation.SurfaceWithEdges,
                        },
                    ],
                ),
                item_title="text",
                item_value="value",
                label="Representation",
                hide_details=True,
                density="compact",
                outlined=True,
                classes="pt-1",
            )
            if show_color:
                with v3.VRow(classes="pt-2", density="compact", no_gutters=True):
                    with v3.VCol(classes="pr-1", cols="6"):
                        # Color By
                        v3.VSelect(
                            label="Color by",
                            v_model=(
                                f"mesh_color_field_{ui_id}",
                                sc.vtk_color_fields[0]["value"],
                            ),
                            items=("color_field_list", sc.vtk_color_fields),
                            item_title="text",
                            item_value="value",
                            hide_details=True,
                            density="compact",
                            outlined=True,
                            classes="pt-1",
                        )
                    with v3.VCol(classes="pl-1", cols="6"):
                        v3.VSelect(
                            v_show=(f"mesh_color_is_vector_{ui_id}",),
                            label="Component",
                            v_model=(
                                f"mesh_color_component_{ui_id}",
                                VectorComponent.Magnitude,
                            ),
                            items=("component_list", VECTOR_COMPONENTS),
                            item_title="text",
                            item_value="value",
                            hide_details=True,
                            density="compact",
                            outlined=True,
                            classes="pt-1",
                        )
                with v3.VRow(classes="pt-2", density="compact", no_gutters=True):
                    with v3.VCol(cols="6"):
                        # Color Map
                        v3.VSelect(
                            label="Colormap",
                            v_model=(f"mesh_color_preset_{ui_id}", 0),
                            items=(
                                "colormaps",
                                [
                                    {"text": lut.name, "value": i}
                                    for i, lut in enumerate(lookup_tables)
                                ],
                            ),
                            item_title="text",
                            item_value="value",
                            hide_details=True,
                            density="compact",
                            outlined=True,
                            classes="pt-1",
                        )
            v3.VSlider(
                # Opacity
                v_model=(f"mesh_opacity_{ui_id}", 1.0),
                min=0,
                max=1,
                step=0.1,
                label="Opacity",
                classes="mt-1",
                hide_details=True,
                density="compact",
            )
            if show_streamlines and sc.vtk_vector_fields and sc.vtk_seed_surfaces:
                v3.VDivider(classes="my-3")
                v3.VCheckbox(
                    v_model=("streamlines_enabled", False),
                    label="Show streamlines",
                    hide_details=True,
                    density="compact",
                )
                with v3.VSheet(
                    v_show=("streamlines_enabled",),
                    classes="pl-3 ml-1",
                    style="border-left: 2px solid rgba(128,128,128,0.35);",
                    elevation=0,
                ):
                    with v3.VRow(classes="pt-1", density="compact", no_gutters=True):
                        with v3.VCol(cols="6"):
                            v3.VSelect(
                                label="Vector field",
                                v_model=("streamlines_field",),
                                items=("vector_field_list", sc.vtk_vector_fields),
                                item_title="text",
                                item_value="value",
                                hide_details=True,
                                density="compact",
                                outlined=True,
                                classes="pt-1",
                            )
                        with v3.VCol(cols="6"):
                            v3.VSelect(
                                label="Seed surface",
                                v_model=("streamlines_seed_surface",),
                                items=("seed_surface_list", sc.vtk_seed_surfaces),
                                item_title="text",
                                item_value="value",
                                hide_details=True,
                                density="compact",
                                outlined=True,
                                classes="pt-1",
                            )
                    v3.VSlider(
                        label="Seeds",
                        v_model=("streamlines_n_seeds", STREAMLINE_DEFAULT_SEEDS),
                        min=10,
                        max=STREAMLINE_MAX_SEEDS,
                        step=10,
                        thumb_label=True,
                        hide_details=True,
                        density="compact",
                        classes="mt-3",
                    )
                    v3.VLabel(
                        "{{ streamlines_seed_hint }}",
                        classes="text-caption text-medium-emphasis",
                    )
            if show_slice:
                v3.VDivider(classes="my-3")
                v3.VSelect(
                    label="Slice / Clip",
                    v_model=("slice_mode",),
                    items=("slice_modes", SLICE_MODES),
                    item_title="text",
                    item_value="value",
                    hide_details=True,
                    density="compact",
                    outlined=True,
                )
                with v3.VSheet(
                    v_show="slice_mode !== 'off'",
                    classes="pl-3 ml-1 pt-2",
                    style="border-left: 2px solid rgba(128,128,128,0.35);",
                    elevation=0,
                ):
                    v3.VSelect(
                        label="Normal",
                        v_model=("slice_axis",),
                        items=("slice_axes_list", list(AXIS_NORMALS.keys())),
                        hide_details=True,
                        density="compact",
                        outlined=True,
                        classes="pt-1",
                    )
                    with v3.VRow(
                        v_show="slice_axis === 'Custom'",
                        classes="pt-2",
                        no_gutters=True,
                    ):
                        for lbl, var in (
                            ("nx", "slice_nx"),
                            ("ny", "slice_ny"),
                            ("nz", "slice_nz"),
                        ):
                            with v3.VCol(cols="4", classes="px-1"):
                                v3.VTextField(
                                    label=lbl,
                                    v_model_number=(var,),
                                    type="number",
                                    step="0.1",
                                    hide_details=True,
                                    density="compact",
                                )
                    v3.VSlider(
                        label="Position",
                        v_model=("slice_offset",),
                        min=-1.0,
                        max=1.0,
                        step=0.01,
                        thumb_label=True,
                        hide_details=True,
                        density="compact",
                        classes="mt-3",
                    )
                    v3.VCheckbox(
                        v_show="slice_mode === 'clip'",
                        v_model=("slice_invert",),
                        label="Invert clip side",
                        hide_details=True,
                        density="compact",
                    )

            callback = self._make_representation_callback(mesh_id)
            self.state.change(f"mesh_representation_{ui_id}")(callback)
            callback = self._make_color_field_callback(mesh_id)
            self.state.change(f"mesh_color_field_{ui_id}")(callback)
            self.state.change(f"mesh_color_component_{ui_id}")(callback)
            callback = self._make_color_preset_callback(mesh_id)
            self.state.change(f"mesh_color_preset_{ui_id}")(callback)
            callback = self._make_opacity_callback(mesh_id)
            self.state.change(f"mesh_opacity_{ui_id}")(callback)
