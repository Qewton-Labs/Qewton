# Algorithms Submodule

The `algorithms` module contains the library of computational building blocks used to construct models. It provides standardized implementations of mathematical, statistical, and neural operations. Is also planned to include more complex algorithms such as Fourier Neural Operators, PCA-Nets, DeepONets, PDE solvers and more.s

## Key Components
- **`Building Blocks`**:
    - **Math**: Basic functions.
    - **Matrix Operations**: MatMul, SVD, and linear algebra.
    - **Statistics**: Reduction operations like Mean, Sum, and Std with support for keep-dims logic.
    - **Reshaping**: Flatten, Transpose, and Slicing operations.

## Goals
1. **Portability**: Write a model once (using the building blocks) and run it on any supported backend. Same operations across all frameworks.
2. **Extensibility**: Provide a clear template for users to wrap their own custom operations or library functions.
3. **Efficiency**: Use fast implementations of backends but combine them in a user-friendly way.

## Implementation Strategy
Subclasses of `Node` typically implement a `forward` method that delegates to `self.implementation`. This allows for a default implementation while providing hooks for framework-specific optimizations when needed. At execution time, the graph will call `forward`, which will use the correct implementation based on the active backend.

## Writing a node
- Declare the configuration of every port in the annotations of `forward`; use one object (e.g. an `EllipsisAxes`) in several annotations for axes the node passes through.
- Implement `relate` only if the node changes dims by their index (see `qewton.config`).
- Make sizes that fix how the node is built `HyperParameter`s and bind the dims of the ports to them (`AxesDim(hp)`); accept None for input sizes inferred from the data (`HyperParameter.size_from`).
- Create weights in `build()`; `setup()` belongs to the library and must not be overridden.