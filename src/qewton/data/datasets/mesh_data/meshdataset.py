from typing import Any

from qewton.config.variables import Variable
from qewton.geometries.discrete.meshes.mesh import Mesh
from qewton.data.datasets.base import DataSet


class MeshDataset(DataSet):
    def __init__(
        self,
        meshes: list[Mesh],
        data: list[list[Any]],
        feature_variables: list[Variable],
        geometry_variable: Variable | None = None,
    ):
        self.meshes = meshes
        self.data = data
        self.feature_variables = feature_variables
        self.geometry_variable = geometry_variable

        # checking data
        if len(meshes) == 0:
            raise ValueError("There are no meshes.")

        if len(data) != len(feature_variables):
            raise ValueError("Data and feature variables differ in length.")

        for i in range(len(feature_variables)):
            if len(meshes) != len(data[i]):
                raise ValueError("Variable doesn't have data for all meshes.")
