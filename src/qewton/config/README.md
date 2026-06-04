# Config Submodule

The `config` module handles the description of data shapes and descriptions within the graph. It allows for automatic inference and checking of tensor shapes, as well as the semantic meaning of different axes (e.g., batch, feature, spatial). This is crucial for ensuring that the computational graph is consistent and that operations are applied to compatible tensors. In addition, it simplifies methods such as plotting, since variables can be automatically inferred.

Example: `DataConfiguration(BatchAxes(...), GeometryAxes(image_grid), FeatureAxes(Variable('f', 1)))` describes a tensor with arbitrary batch dimension(s), spatial dimensions corresponding to an image grid (e.g. 256x256), and a feature dimension function values on the image grid.

## Key Components
- **`DataConfiguration`**: Encapsulates the expected structure of a tensor at a specific port, including its axes and dimensionality. Allows to include symbolic or non-specified dimensions (e.g., `EllipsisDim`) that can be inferred later.
- **`Axes` & `AxesDim`**: Provide a rich type system for dimensions. There are three basic types of axes:
    - `BatchAxes`: For sample/batch dimensions.
    - `GeometryAxes`: For spatial/temporal coordinates. Is connected to a (possibly dummy) `Geometry` object that defines the underlying structure and variables.
    - `FeatureAxes`: For channels or physical variables. Their names can be inferred from previously defined `Variable` objects. Every `DataConfiguration` can contain at most one `FeatureAxes` object.
- **Unification System**: Compares and merges configurations from different nodes to ensure shape compatibility and infer missing dimensions (e.g., handling `EllipsisDim`).
- **`Backend`**: Abstracts framework-specific details, allowing the same logic to target PyTorch or TensorFlow. This includes the data types such as torch.Tensor or tf.Tensor.

## Goals
1. **Strong Typing for Tensors**: Move beyond simple shape tuples to semantically meaningful axes.
2. **Automatic Shape Inference**: Automatically propagate dimension sizes across the graph, reducing manual configuration.
3. **Backend Agnosticism**: Provide a unified interface for tensor operations across different deep learning libraries.
4. **Error Prevention**: Catch dimension mismatches or incompatible variable mappings at graph-build time rather than at runtime.
5. **Enhanced Visualization**: Use the configuration information to automatically generate plots.

## Axis Logic
The module supports symbolic relationships between dimensions, such as `AddedDim` or `ProductDim`, allowing the system to understand that an output dimension is the sum or product of specific inputs.

## TODO:
- could we simplify the config passing to use common AxesDim objects nearly around the whole graph? this would allow for less passing operations and less objects.