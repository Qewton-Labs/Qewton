import qewton
from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.spec import PlotSpec, SliderSpec, SelectorSpec

# TRAME
from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageLayout, SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
from trame.widgets import plotly, vtk, trame
from trame.decorators import change

from trame_vtk.modules.vtk.serializers import configure_serializer

# VTK
from vtkmodules.vtkCommonCore import vtkPoints, vtkLookupTable
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkUnstructuredGrid,
    vtkDataObject,
    vtkPiecewiseFunction,
    vtkPolyData,
    VTK_TETRA,
    VTK_TRIANGLE,
)

# from vtkmodules.vtkRenderingVolume import (
#    vtkProjectedTetrahedraMapper
# )
# import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingVolumeOpenGL2 import (
    vtkOpenGLProjectedTetrahedraMapper,
)
from vtkmodules.vtkFiltersCore import vtkContourFilter

# from vtkmodules.vtkFiltersSources import vtkConeSource
# from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
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
    vtkVolume,
    vtkVolumeProperty,
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
)

# -----------------------------------------------------------------------------
# VTK STUFF
# -----------------------------------------------------------------------------


def create_mesh(vertices, cells, cell_type):
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)

    n_vertices_per_cell = cells.shape[1]
    n_cells = cells.shape[0]

    # Points shared by the mesh
    points = vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))

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
        mesh.SetPoints(points)
        mesh.SetCells(VTK_TETRA, vtk_cells)
    elif cell_type == VTK_TRIANGLE:
        mesh = vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetPolys(vtk_cells)
    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")

    return mesh


# Configure scene encoder
# configure_serializer(encode_lut=True, skip_light=True)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


class Representation:
    Points = 0
    Wireframe = 1
    Surface = 2
    SurfaceWithEdges = 3


class LookupTable:
    Rainbow = 0
    Inverted_Rainbow = 1
    Greyscale = 2
    Inverted_Greyscale = 3


VECTOR_COMPONENTS = [
    {"text": "Magnitude", "value": "Magnitude"},
    {"text": "X", "value": "X"},
    {"text": "Y", "value": "Y"},
    {"text": "Z", "value": "Z"},
]
COMPONENT_INDEX = {"X": 0, "Y": 1, "Z": 2, "Magnitude": -1}


# -----------------------------------------------------------------------------
# VTK pipeline
# -----------------------------------------------------------------------------

renderer = vtkRenderer()
renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

# Enable polygon offset for coincident geometry
# vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

# Tune the offset if needed
# vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(0.0, 0.0)

# Cone

# cone_source = vtkConeSource()
# mapper = vtkPolyDataMapper()
# mapper.SetInputConnection(cone_source.GetOutputPort())
# actor = vtkActor()
# actor.SetMapper(mapper)
# actor.GetProperty().SetColor(1.0, 0.0, 0.0)
# actor.GetProperty().SetOpacity(0.5)
# actor.SetPosition(2.0, 0.0, 0.0)
# renderer.AddActor(actor)
#
# mesh_actor = actor

# Read Data
if False:
    reader = vtkXMLUnstructuredGridReader()
    # filename = os.path.join(CURRENT_DIRECTORY, "trame-data/disk_out_ref.vtu")
    # print("cwd:", CURRENT_DIRECTORY)
    filename = "src/qewton/visualization/trame-data/disk_out_ref.vtu"
    print("filename:", filename)
    reader.SetFileName(filename)
    reader.Update()

    # Extract Array/Field information
    dataset_arrays = []
    fields = [
        (reader.GetOutput().GetPointData(), vtkDataObject.FIELD_ASSOCIATION_POINTS),
        (reader.GetOutput().GetCellData(), vtkDataObject.FIELD_ASSOCIATION_CELLS),
    ]
    for field in fields:
        field_arrays, association = field
        for i in range(field_arrays.GetNumberOfArrays()):
            array = field_arrays.GetArray(i)
            array_range = array.GetRange()
            dataset_arrays.append(
                {
                    "text": array.GetName(),
                    "value": i,
                    "range": list(array_range),
                    "type": association,
                }
            )
    default_array = dataset_arrays[0]
    default_min, default_max = default_array.get("range")

    # Mesh
    mesh_mapper = vtkDataSetMapper()
    mesh_mapper.SetInputConnection(reader.GetOutputPort())
    mesh_actor = vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    renderer.AddActor(mesh_actor)

    # Mesh: Setup default representation to surface
    mesh_actor.GetProperty().SetRepresentationToSurface()
    mesh_actor.GetProperty().SetPointSize(1)
    mesh_actor.GetProperty().EdgeVisibilityOff()

    # Mesh: Apply rainbow color map
    mesh_lut = mesh_mapper.GetLookupTable()
    mesh_lut.SetHueRange(0.666, 0.0)
    mesh_lut.SetSaturationRange(1.0, 1.0)
    mesh_lut.SetValueRange(1.0, 1.0)
    mesh_lut.Build()

    # Mesh: Color by default array
    mesh_mapper.SelectColorArray(default_array.get("text"))
    mesh_mapper.GetLookupTable().SetRange(default_min, default_max)
    if default_array.get("type") == vtkDataObject.FIELD_ASSOCIATION_POINTS:
        mesh_mapper.SetScalarModeToUsePointFieldData()
    else:
        mesh_mapper.SetScalarModeToUseCellFieldData()
    mesh_mapper.SetScalarVisibility(True)
    mesh_mapper.SetUseLookupTableScalarRange(True)

    # Contour
    # contour = vtkContourFilter()
    # contour.SetInputConnection(reader.GetOutputPort())
    # contour_mapper = vtkDataSetMapper()
    # contour_mapper.SetInputConnection(contour.GetOutputPort())
    # contour_actor = vtkActor()
    # contour_actor.SetMapper(contour_mapper)
    # renderer.AddActor(contour_actor)
    #
    ## Contour: ContourBy default array
    # contour_value = 0.5 * (default_max + default_min)
    # contour.SetInputArrayToProcess(
    #    0, 0, 0, default_array.get("type"), default_array.get("text")
    # )
    # contour.SetValue(0, contour_value)
    #
    ## Contour: Setup default representation to surface
    # contour_actor.GetProperty().SetRepresentationToSurface()
    # contour_actor.GetProperty().SetPointSize(1)
    # contour_actor.GetProperty().EdgeVisibilityOff()
    #
    ## Contour: Apply rainbow color map
    # contour_lut = contour_mapper.GetLookupTable()
    # contour_lut.SetHueRange(0.666, 0.0)
    # contour_lut.SetSaturationRange(1.0, 1.0)
    # contour_lut.SetValueRange(1.0, 1.0)
    # contour_lut.Build()
    #
    ## Contour: Color by default array
    # contour_mapper.SelectColorArray(default_array.get("text"))
    # contour_mapper.GetLookupTable().SetRange(default_min, default_max)
    # if default_array.get("type") == vtkDataObject.FIELD_ASSOCIATION_POINTS:
    #    contour_mapper.SetScalarModeToUsePointFieldData()
    # else:
    #    contour_mapper.SetScalarModeToUseCellFieldData()
    # contour_mapper.SetScalarVisibility(True)
    # contour_mapper.SetUseLookupTableScalarRange(True)

# Cube Axes
cube_axes = vtkCubeAxesActor()
renderer.AddActor(cube_axes)

# Cube Axes: Boundaries, camera, and styling
# cube_axes.SetBounds(mesh_actor.GetBounds())
cube_axes.SetCamera(renderer.GetActiveCamera())
cube_axes.SetXLabelFormat("%6.1f")
cube_axes.SetYLabelFormat("%6.1f")
cube_axes.SetZLabelFormat("%6.1f")
cube_axes.SetFlyModeToOuterEdges()

renderer.ResetCamera()

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
def color_by_array(actor, mesh, array):
    _min, _max = array.get("range")
    name = array.get("text")
    mapper = actor.GetMapper()
    mapper.SelectColorArray(name)
    # mesh.GetPointData().SetActiveScalars(name)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SetScalarRange(_min, _max)
    mapper.ScalarVisibilityOn()


def color_by_field(actor, mesh, field, component=None):
    mapper = actor.GetMapper()
    name = field["text"]
    lut = mapper.GetLookupTable()

    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(name)
    if field["type"] == "scalar":
        print("Color by", name, "range:", *field["range"])
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
        print("Color by", name, ",", comp, "range:", *field["component_ranges"][comp])
    mapper.ScalarVisibilityOn()


# Color Map Callback
def use_preset(actor, preset):
    lut = actor.GetMapper().GetLookupTable()
    if preset == LookupTable.Rainbow:
        lut.SetHueRange(0.666, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif preset == LookupTable.Inverted_Rainbow:
        lut.SetHueRange(0.0, 0.666)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif preset == LookupTable.Greyscale:
        lut.SetHueRange(0.0, 0.0)
        lut.SetSaturationRange(0.0, 0.0)
        lut.SetValueRange(0.0, 1.0)
    elif preset == LookupTable.Inverted_Greyscale:
        lut.SetHueRange(0.0, 0.666)
        lut.SetSaturationRange(0.0, 0.0)
        lut.SetValueRange(1.0, 0.0)
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
        """Builds a Trame app displaying `mesh`"""
        app = TrameApplication()

        app._build_mesh_ui(mesh, scalar_fields, vector_fields)
        return app

    def _hide_kitware_link(self):
        # Inject CSS to hide ONLY the Kitware website link inside the footer
        from trame.widgets import client

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

            # with self.ui.toolbar:
            #    v3.VSpacer()
            #    v3.VSelect(
            #        v_model=("active_plot_name", title),
            #        items=("plots", [title]),
            #        hide_details=True,
            #        density="compact",
            #        style="max-width: 200px;",
            #        variant="outlined",
            #        classes="mr-4",
            #    )

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
                    # view = vtk.VtkRemoteView(renderWindow, interactive_ratio=1)
                    view = vtk.VtkLocalView(renderWindow)
                    # view = vtk.VtkRemoteLocalView(
                    #    self._renderWindow, namespace="view", mode="local", interactive_ratio=1
                    # )
                    self.ctrl.view_update = view.update
                    self.ctrl.view_reset_camera = view.reset_camera

            with self.ui.drawer as drawer:
                # drawer components
                drawer.width = 350
                self.pipeline_widget()
                v3.VDivider(classes="mb-2")
                for i, m in enumerate(self.vtk_meshes):
                    self.mesh_card(m[0], i)

    def _init_vtk(self):
        # self._renderer = vtkRenderer()
        # self._renderWindow = vtkRenderWindow()
        # self._renderWindow.AddRenderer(self._renderer)
        self._renderWindow = renderWindow

    def _init_mesh(self, mesh, scalar_fields, vector_fields):
        # self._init_cone_mesh()
        self._init_unstructured_grid(mesh, scalar_fields, vector_fields)

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

    def _init_unstructured_grid(
        self, qmesh: "qewton.geometries.MeshGeometry", scalar_fields, vector_fields
    ):
        tet_mesh = create_mesh(qmesh.mesh.vertices, qmesh.mesh.cells, VTK_TETRA)
        self._init_scalar_fields(tet_mesh, scalar_fields)
        self._init_vector_fields(tet_mesh, vector_fields)
        self.vtk_color_fields = self.vtk_scalar_fields + self.vtk_vector_fields

        tet_mapper = vtkDataSetMapper()
        tet_mapper.SetInputData(tet_mesh)
        tet_actor = vtkActor()
        tet_actor.SetMapper(tet_mapper)
        # actor.GetProperty().SetRepresentationToWireframe()
        # tet_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)
        # tet_mapper.SetColorModeToMapScalars()
        # tet_mapper.ScalarVisibilityOn()
        # tet_mapper.SetScalarRange(mi, ma)
        # lut = vtkLookupTable()
        # lut.SetNumberOfTableValues(256)
        # lut.SetHueRange(0.667, 0.0)  # Blue to red
        # lut.SetRange(mi, ma)
        # lut.Build()
        # tet_mapper.SetLookupTable(lut)
        renderer.AddActor(tet_actor)
        cube_axes.SetBounds(tet_actor.GetBounds())

        self.vtk_meshes = [("Mesh", tet_actor, tet_mapper, tet_mesh)]

        # TODO: share vtkPoints among meshes
        submeshes = [
            name for name, value in qmesh.mesh.marker_labels.items() if value[1] == 2
        ]
        # submeshes = []
        # submeshes = submeshes[:1]
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
            # surface_actor.GetProperty().SetOpacity(0.5)
            # surface_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1.0, -1.0)
            renderer.AddActor(surface_actor)
            self.vtk_meshes.append(
                (surf_name, surface_actor, surface_mapper, surface_mesh)
            )

    def _init_cone_mesh(self):
        cone_source = vtkConeSource()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cone_source.GetOutputPort())
        mesh_actor = vtkActor()
        mesh_actor.SetMapper(mapper)
        mesh_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        mesh_actor.GetProperty().SetOpacity(0.5)
        mesh_actor.SetPosition(2.0, 0.0, 0.0)
        renderer.AddActor(mesh_actor)
        cube_axes.SetBounds(mesh_actor.GetBounds())

    # -----------------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------------

    @change("cube_axes_visibility")
    def update_cube_axes_visibility(self, cube_axes_visibility, **_):
        cube_axes.SetVisibility(cube_axes_visibility)
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

        self.vtk_meshes[idx][1].SetVisibility(visibility)
        self.ctrl.view_update()

    def update_mesh_representation(self, mesh_id, mesh_representation, **_):
        mesh_actor = self.vtk_meshes[mesh_id][1]
        update_representation(mesh_actor, mesh_representation)
        self.ctrl.view_update()

    def update_mesh_opacity(self, mesh_id, mesh_opacity, **_):
        mesh_actor = self.vtk_meshes[mesh_id][1]
        mesh_actor.GetProperty().SetOpacity(mesh_opacity)
        self.ctrl.view_update()

    def update_mesh_color_preset(self, mesh_id, mesh_color_preset, **_):
        mesh_actor = self.vtk_meshes[mesh_id][1]
        use_preset(mesh_actor, mesh_color_preset)
        self.ctrl.view_update()

    def update_mesh_color_array_(self, mesh_id, mesh_color_array_idx, **_):
        array = self.vtk_scalar_fields[mesh_color_array_idx]
        mesh_actor = self.vtk_meshes[mesh_id][1]
        mesh = self.vtk_meshes[mesh_id][3]
        color_by_array(mesh_actor, mesh, array)
        self.ctrl.view_update()

    def update_mesh_color_array(
        self, mesh_id, mesh_color_field, mesh_color_component=None, **_
    ):
        mesh_actor = self.vtk_meshes[mesh_id][1]
        mesh = self.vtk_meshes[mesh_id][3]
        color_by_field(mesh_actor, mesh, mesh_color_field, mesh_color_component)
        self.ctrl.view_update()

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
            # p = {"id": str(i+1), "parent": "0" if i == 0 else "1", "visible": 1, "name": m[0]}
            p = {"id": str(i + 1), "parent": str(i), "visible": 1, "name": m[0]}
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

    def mesh_card(self, title, mesh_id):
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
                        {"text": "Points", "value": 0},
                        {"text": "Wireframe", "value": 1},
                        {"text": "Surface", "value": 2},
                        {"text": "SurfaceWithEdges", "value": 3},
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
                        v_model=(f"mesh_color_preset_{ui_id}", LookupTable.Rainbow),
                        items=(
                            "colormaps",
                            [
                                {"text": "Rainbow", "value": 0},
                                {"text": "Inv Rainbow", "value": 1},
                                {"text": "Greyscale", "value": 2},
                                {"text": "Inv Greyscale", "value": 3},
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

            callback = self._make_representation_callback(mesh_id)
            self.state.change(f"mesh_representation_{ui_id}")(callback)
            callback = self._make_color_field_callback(mesh_id)
            self.state.change(f"mesh_color_field_{ui_id}")(callback)
            self.state.change(f"mesh_color_component_{ui_id}")(callback)
            callback = self._make_color_preset_callback(mesh_id)
            self.state.change(f"mesh_color_preset_{ui_id}")(callback)
            callback = self._make_opacity_callback(mesh_id)
            self.state.change(f"mesh_opacity_{ui_id}")(callback)
