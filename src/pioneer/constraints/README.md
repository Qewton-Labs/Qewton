# Constraints Submodule

The `constraints` module defines the objectives and metrics used to guide and log the optimization process. Constraints are specialized nodes that evaluate model outputs against specific criteria, such as ground truth data or physical laws. Enables flexible combination of different types of constraints (e.g., data-fitting losses, physics-based residuals, or monitoring metrics) within the same graph.

## Integration
During training, the `GraphBasedTrainer` scans the graph for `Constraint` nodes. It uses their `evaluated_in_mode` to seperate training and validation execution. Constraints which should be optimized need to be passed to the `Trainer` as objectives.