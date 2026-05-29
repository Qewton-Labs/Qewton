# Graphs Submodule

The `graphs` module provides the core infrastructure for building and executing Directed Acyclic Graphs (DAGs) in Qewton. It allows users to define modular computation pipelines where each operation is encapsulated as a `Node`.

## Key Components
- **`Graph`**: The main container that manages nodes, edges, and topological sorting (using Kahn's algorithm) for execution. It supports programmatic construction through the `tracker` context manager, to avoid large amounts of manual calls to `graph.connect()`.
- **`Node`**: The base class for all functional blocks. Nodes define input and output ports and can represent anything from simple math to entire subgraphs. While multiple edges can connect to a single output port, each input port can only have one incoming edge.
Data loaders and constraints are also implemented as nodes, allowing them to be integrated into the graph execution.
- **`Edge`**: Manages the data flow (data can be of any type) between ports, including support for connections to outside the local graph context (e.g., in a `GraphNode`) and skip connections.
- **`GraphNode`**: Enables hierarchical nesting by encapsulating an entire graph within a single node.
- **`TrackingObject`**: Allows users to write standard Python functions that are automatically converted into graph structures. Note: This does not yet support control flow constructs like loops or conditionals.
- **Pipelines**: Predefined graph templates for common tasks (e.g., PINN training) that can be easily customized and extended.

## Goals
1. **Modularity**: Break complex physics models into reusable, atomic components.
2. **Implicit Construction**: Allow users to build graphs using standard Python syntax and operators.
3. **Validation**: Ensure that connections between nodes are compatible before execution.
4. **Flexibility**: Support both simple sequential execution and complex, multi-input/multi-output DAGs. Can therefore cover arbitrary algorithms, e.g. for time-dependent problems, multi-physics coupling, or multi-task learning.
5. **Hierarchical Nesting**: Enable users to create subgraphs that can be reused as single nodes in larger graphs.
6. **Independence of Backend**: Does not only allow for the use of multiple deep learning frameworks, but also for non-deep learning computations, e.g. for data preprocessing or classical numerical methods.
