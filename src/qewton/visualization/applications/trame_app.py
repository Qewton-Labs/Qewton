import qewton
from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.spec import PlotSpec, SliderSpec, SelectorSpec
from enum import IntEnum
from dataclasses import dataclass

# TRAME
from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageLayout, SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
from trame.widgets import plotly, vtk, trame
from trame.decorators import change
from trame.widgets import client

# VTK
from vtkmodules.vtkCommonCore import vtkPoints, vtkLookupTable
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkCellLocator,
    vtkUnstructuredGrid,
    vtkPolyData,
    VTK_TETRA,
    VTK_TRIANGLE,
)

from vtkmodules.vtkFiltersCore import (
    vtkCellCenters,
    vtkPolyDataNormals,
    vtkTubeFilter,
)
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor

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

VECTOR_COMPONENTS = [
    {"text": "Magnitude", "value": "Magnitude"},
    {"text": "X", "value": "X"},
    {"text": "Y", "value": "Y"},
    {"text": "Z", "value": "Z"},
]
COMPONENT_INDEX = {"X": 0, "Y": 1, "Z": 2, "Magnitude": -1}

STREAMLINE_SEED_OFFSET_FACTOR = 1e-4  # relative to bounding box diagonal
STREAMLINE_MAX_SEEDS = 500
STREAMLINE_DEFAULT_SEEDS = 100

STREAMLINE_TUBE_RADIUS_FACTOR = 1.5e-3  # relative to the bounding box diagonal
STREAMLINE_TUBE_SIDES = 24

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


class Streamlines:
    def __init__(self):
        self.enabled = False
        self.actor: vtkActor | None = None
        self.mapper: vtkMapper | None = None
        self.tracer: vtkStreamTracer | None = None
        self.tube: vtkTubeFilter | None = None
        self.seed_polydata = None
        self.vector_field = None
        self.seed_surface_id: int | None = None
        self.n_seeds = STREAMLINE_DEFAULT_SEEDS
        self.n_available = 0

    def build(self, volume_mesh):
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

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
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
        self.mapper = mapper
        self.actor = actor


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


def init_scene():
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

    # Cube Axes
    cubeAxes = vtkCubeAxesActor()
    renderer.AddActor(cubeAxes)

    # Cube Axes: camera, and styling
    cubeAxes.SetCamera(renderer.GetActiveCamera())
    cubeAxes.SetXLabelFormat("%6.1f")
    cubeAxes.SetYLabelFormat("%6.1f")
    cubeAxes.SetZLabelFormat("%6.1f")
    cubeAxes.SetFlyModeToOuterEdges()

    renderer.ResetCamera()
    return renderer, renderWindow, cubeAxes


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


# Color By Callback
def color_by_field(vtk_item, field, component=None):
    mapper = vtk_item.mapper
    name = field["text"]
    lut = mapper.GetLookupTable()

    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(name)
    if field["type"] == "scalar":
        mapper.SetScalarRange(*field["range"])
        lut.SetVectorModeToComponent()
        lut.SetVectorComponent(0)
    else:
        comp = component or "Magnitude"
        value_range = field["component_ranges"][comp]
        mapper.SetScalarRange(*value_range)
        if comp == "Magnitude":
            lut.SetVectorModeToMagnitude()
        else:
            lut.SetVectorModeToComponent()
            lut.SetVectorComponent(COMPONENT_INDEX[comp])
        mapper.ColorByArrayComponent(name, COMPONENT_INDEX[comp])
    mapper.ScalarVisibilityOn()


# Color Map Callback
def use_preset(actor, preset):
    lut = actor.GetMapper().GetLookupTable()
    table = lookup_tables[preset]
    lut.SetHueRange(*table.hue_range)
    lut.SetSaturationRange(*table.saturation_range)
    lut.SetValueRange(*table.value_range)
    lut.Build()


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

        with SinglePageWithDrawerLayout(
            self.server, theme=("theme", "light"), hide_footer=True
        ) as self.ui:
            self.state.vtk_field_types = {
                f["value"]: f["type"] for f in self.vtk_color_fields
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
                for i, m in enumerate(self.vtk_meshes):
                    vol_mesh = i == 0
                    self.mesh_card(
                        m.name, i, allow_color=vol_mesh, allow_streamlines=vol_mesh
                    )

    def _init_vtk(self):
        self.renderer, self.renderWindow, self.cubeAxes = init_scene()

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
                "X": [values[:, 0].min(), values[:, 0].max()],
                "Y": [values[:, 1].min(), values[:, 1].max()],
                "Z": [values[:, 2].min(), values[:, 2].max()] if dim == 3 else None,
                "Magnitude": [magnitudes.min(), magnitudes.max()],
            }

            self.vtk_vector_fields.append(
                {
                    "text": name,
                    "value": name,
                    "type": "vector",
                    "component_ranges": component_ranges,
                }
            )

    def _init_streamlines(self):
        self.vtk_streamlines = Streamlines()

        # Boundary surfaces for seed selector
        self.vtk_seed_surfaces = [
            {"text": mesh.name, "value": idx}
            for idx, mesh in enumerate(self.vtk_meshes)
            if idx != 0  # Don't include volume mesh itself
        ]

        self.state.streamlines_enabled = False
        self.state.streamlines_field = (
            self.vtk_vector_fields[0]["value"] if self.vtk_vector_fields else None
        )
        self.state.streamlines_seed_surface = (
            self.vtk_seed_surfaces[0]["value"] if self.vtk_seed_surfaces else None
        )
        self.state.streamlines_n_seeds = STREAMLINE_DEFAULT_SEEDS
        self.state.streamlines_seed_hint = ""

    def _build_streamlines(self):
        """Create Tracer/Mapper/Actor once upon first use."""
        sl = self.vtk_streamlines
        if sl.tracer is not None:
            return

        volume_mesh = self.vtk_meshes[0].mesh
        sl.build(volume_mesh)
        assert sl.actor
        self.renderer.AddActor(sl.actor)

    def _update_streamlines(self):
        sl = self.vtk_streamlines

        if not self.state.streamlines_enabled:
            if sl.actor is not None:
                sl.actor.SetVisibility(False)
            self.ctrl.view_update()
            return

        self._build_streamlines()

        field = self.state.streamlines_field
        surface_id = self.state.streamlines_seed_surface
        n_requested = self.state.streamlines_n_seeds

        if field is None or surface_id is None:
            sl.actor.SetVisibility(False)
            self.ctrl.view_update()
            return

        # Set active vector field for volume mesh
        volume_mesh = self.vtk_meshes[0].mesh
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
            self.state.streamlines_seed_hint = "No seeds available"
            self.ctrl.view_update()
            return

        self.state.streamlines_seed_hint = f"{n_used} seeds from {n_available} faces"

        tracer = sl.tracer
        tracer.SetSourceData(seed_pd)
        tracer.Update()
        sl.tube.Update()

        # Color according to vector field magnitude
        mapper = sl.mapper
        volume_mapper = self.vtk_meshes[0].mapper
        lut_src = volume_mapper.GetLookupTable()
        lut = vtkLookupTable()
        lut.DeepCopy(lut_src)
        lut.SetVectorModeToMagnitude()
        lut.Build()
        out = tracer.GetOutput()
        arr = out.GetPointData().GetArray(field)
        if arr is not None:
            lut = mapper.GetLookupTable()
            lut.SetVectorModeToMagnitude()
            lut.Build()
            mapper.SelectColorArray(field)
            mapper.SetScalarRange(arr.GetRange(-1))
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
        self._sync_streamline_colors()

        sl.actor.SetVisibility(True)
        self.ctrl.view_update()

    def _sync_streamline_colors(self):
        """Apply the volume mesh color mapping to the streamline mapper."""
        sl = self.vtk_streamlines
        if sl.mapper is None:
            return

        field = sl.vector_field
        if field is None:
            return

        volume_mapper = self.vtk_meshes[0].mapper
        lut = volume_mapper.GetLookupTable()
        if lut is None:
            return

        sl.mapper.SetLookupTable(lut)
        sl.mapper.SelectColorArray(field)
        sl.mapper.SetScalarRange(volume_mapper.GetScalarRange())
        sl.mapper.ScalarVisibilityOn()

    def _init_mesh(
        self, qmesh: "qewton.geometries.MeshGeometry", scalar_fields, vector_fields
    ):
        tet_mesh = create_mesh(qmesh.mesh.vertices, qmesh.mesh.cells, VTK_TETRA)
        self._init_scalar_fields(tet_mesh, scalar_fields)
        self._init_vector_fields(tet_mesh, vector_fields)
        self.vtk_color_fields = self.vtk_scalar_fields + self.vtk_vector_fields
        self._init_meshes(qmesh, tet_mesh)
        self._init_streamlines()

    def _init_meshes(self, qmesh: "qewton.geometries.MeshGeometry", tet_mesh):
        tet_mapper = vtkDataSetMapper()
        tet_mapper.SetInputData(tet_mesh)
        tet_actor = vtkActor()
        tet_actor.SetMapper(tet_mapper)
        self.renderer.AddActor(tet_actor)
        self.cubeAxes.SetBounds(tet_actor.GetBounds())

        self.vtk_meshes = [VtkItem("Mesh", tet_actor, tet_mapper, tet_mesh)]

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
            self.renderer.AddActor(surface_actor)
            self.vtk_meshes.append(
                VtkItem(surf_name, surface_actor, surface_mapper, surface_mesh)
            )

    # -----------------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------------

    @change("cube_axes_visibility")
    def update_cube_axes_visibility(self, cube_axes_visibility, **_):
        self.cubeAxes.SetVisibility(cube_axes_visibility)
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

        self.vtk_meshes[idx].actor.SetVisibility(visibility)
        self.ctrl.view_update()

    def update_mesh_representation(self, mesh_id, mesh_representation, **_):
        mesh_actor = self.vtk_meshes[mesh_id].actor
        update_representation(mesh_actor, mesh_representation)
        self.ctrl.view_update()

    def update_mesh_opacity(self, mesh_id, mesh_opacity, **_):
        mesh_actor = self.vtk_meshes[mesh_id].actor
        mesh_actor.GetProperty().SetOpacity(mesh_opacity)
        self.ctrl.view_update()

    def update_mesh_color_preset(self, mesh_id, mesh_color_preset, **_):
        mesh_actor = self.vtk_meshes[mesh_id].actor
        use_preset(mesh_actor, mesh_color_preset)
        if self.vtk_streamlines.actor is not None and self.state.streamlines_enabled:
            self._sync_streamline_colors()
        self.ctrl.view_update()

    def update_mesh_color_array(
        self, mesh_id, mesh_color_field, mesh_color_component=None, **_
    ):
        color_by_field(self.vtk_meshes[mesh_id], mesh_color_field, mesh_color_component)
        self.ctrl.view_update()

    @change(
        "streamlines_enabled",
        "streamlines_field",
        "streamlines_seed_surface",
        "streamlines_n_seeds",
    )
    def update_streamlines(self, **_):
        self._update_streamlines()

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
        assert len(self.vtk_meshes) >= 1
        pipeline = []
        for i, m in enumerate(self.vtk_meshes):
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

    def _make_color_array_callback(self, mesh_id):
        def on_change(**kwargs):
            ui_id = str(mesh_id + 1)
            value = kwargs[f"mesh_color_array_idx_{ui_id}"]
            self.update_mesh_color_array(mesh_id, value)

        return on_change

    def _make_color_field_callback(self, mesh_id):
        def on_change(**kwargs):
            ui_id = str(mesh_id + 1)
            field_name = kwargs[f"mesh_color_field_{ui_id}"]
            component = kwargs.get(f"mesh_color_component_{ui_id}", None)
            field = next(f for f in self.vtk_color_fields if f["value"] == field_name)
            # Explicitly set state variable to control visibility
            setattr(
                self.state, f"mesh_color_is_vector_{ui_id}", field["type"] == "vector"
            )
            self.update_mesh_color_array(mesh_id, field, component)

        return on_change

    def mesh_card(self, title, mesh_id, allow_color=False, allow_streamlines=False):
        ui_id = str(mesh_id + 1)
        default_field = self.vtk_color_fields[0]
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
            if allow_color:
                with v3.VRow(classes="pt-2", density="compact", no_gutters=True):
                    with v3.VCol(classes="pr-1", cols="6"):
                        # Color By
                        v3.VSelect(
                            label="Color by",
                            v_model=(
                                f"mesh_color_field_{ui_id}",
                                self.vtk_color_fields[0]["value"],
                            ),
                            items=("color_field_list", self.vtk_color_fields),
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
                            v_model=(f"mesh_color_component_{ui_id}", "Magnitude"),
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
            if allow_streamlines and self.vtk_vector_fields and self.vtk_seed_surfaces:
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
                                items=("vector_field_list", self.vtk_vector_fields),
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
                                items=("seed_surface_list", self.vtk_seed_surfaces),
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

            callback = self._make_representation_callback(mesh_id)
            self.state.change(f"mesh_representation_{ui_id}")(callback)
            callback = self._make_color_field_callback(mesh_id)
            self.state.change(f"mesh_color_field_{ui_id}")(callback)
            self.state.change(f"mesh_color_component_{ui_id}")(callback)
            callback = self._make_color_preset_callback(mesh_id)
            self.state.change(f"mesh_color_preset_{ui_id}")(callback)
            callback = self._make_opacity_callback(mesh_id)
            self.state.change(f"mesh_opacity_{ui_id}")(callback)
