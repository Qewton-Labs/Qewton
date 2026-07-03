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

The forward call should also include logic for handling the `DataConfiguration` of the inputs and outputs, ensuring that the shapes and semantics of the tensors are consistent across the graph.