# Data Submodule

Integrates data sources and point samplers into the graph. It provides a standardized way to load and batch data as a source in the computation graph, ensuring that the data is correctly configured and compatible with the model's expectations.

## Key Components
- **`DataNode`**: A specialized `Node` that serves as a source of data in the graph. It can wrap various data sources, such as files, in-memory arrays, or synthetic data generators.
- **`DataSet`**: An abstraction for data containers. For example, `NumpyDataSet` handles NumPy arrays while ensuring they match a specific `DataConfiguration`.
- **`DataLoader`**: A specialized `Node` that loads from a dataset. It handles batching, shuffling, and feeding data into the input ports of subsequent nodes.
- **`PointSampler`**: A node that generates points based on a specified `Geometry` and sampling strategy (e.g., uniform). This is particularly useful for physics-informed models that require collocation points.

## Goals
1. **Interface**: Treat external data sources consistently, regardless of whether they originate from files or geometries.
2. **Graph**: Enable data loading to be a first-class citizen in the DAG, allowing for end-to-end pipelines from raw files to optimized models.
3. **Metadata**: Ensure that the `DataConfiguration` (axes types, semantic meaning) is correctly associated with data from the moment it is loaded.
