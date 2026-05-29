# Optim Submodule

The `optim` module manages the training, optimization, and hyperparameter tuning of graph-based models. It bridges the gap between the static graph structure and the dynamic training process.

## Key Components
- **`Trainer`**: Manages `OptimizationPhases` (which specify optimizers), callbacks (like progress bars or logging), and device placement.
- **`GraphBasedTrainer`**: The standard trainer that extracts constraints (losses/metrics) and parameters directly from the computation graphs.
- **`Tuner`**: A framework for hyperparameter optimization (HPO). It supports parallel trial execution across multiple devices and logs results to CSV by default. Subclasses could also use specific HPO libraries like Optuna or Ray Tune.
- **`HyperParameter`**: Represents tunable values in the graph (e.g., learning rates, layer widths) that are managed by the `Tuner`.

## Goals
1. **Automated Training Pipelines**: Simplify the process of moving from a graph definition to a trained model.
2. **Scalable Tuning**: Provide built-in support for running multiple trials in parallel using multiprocessing.
3. **State Management**: Robustly track training progress, metrics, and hyperparameter snapshots.
4. **Phase-Based Optimization**: Support complex training schedules where different parts of the model or different objectives are optimized in sequence.

## Parallelism
The `Tuner` uses a worker-queue model to distribute training trials across available CPUs or GPUs, ensuring efficient utilization of hardware during hyperparameter searches.