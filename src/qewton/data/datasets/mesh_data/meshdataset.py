from qewton.geometries.discrete.meshes.mesh import Mesh


class MeshData():

    def __init__(self, mesh: Mesh, node_features, edge_features=None, edges=None):
        self.mesh : Mesh = mesh
        self.node_features = node_features
        self.edge_features = edge_features
        self.edges = edges

    @property
    def mesh_key(self):
        


class MeshDataset(Dataset):

    def __init__(self, mesh_data_list):
        self.mesh_data_list = mesh_data_list

        # Batching mesh data
        
        # Compute hierarchical structure for batching meshes


