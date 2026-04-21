from collections import defaultdict, deque
from typing import Any
from itertools import product
import warnings
import random

from .hyperparameter_base import HyperParameter
from .categorical_hyperparameter import CategoricalHyperparameter


class HyperParameterDAG:
    """A directed acyclic graph (DAG) representing HyperParameter dependencies."""

    def __init__(self, hyperparameters: set[HyperParameter]):
        name_list = []
        for hp in hyperparameters:
            if hp.name in name_list:
                raise ValueError(
                    f"Found at least two HyperParameters with the name '{hp.name}'.\
                        Can not uniquely carry out the tuning process."
                )
            name_list.append(hp.name)

        self.graph = self.build_graph(hyperparameters)
        self.sort()

    def build_graph(self, hyperparameters: set[HyperParameter]):
        graph = defaultdict(set)
        for hp in hyperparameters:
            for dep in hp.dependencies:
                # Edge from parameter to parameters that depend on it
                graph[dep].add(hp)
            if hp not in graph:
                graph[hp] = set()
        return graph

    def sort(self):
        in_degree = {node: 0 for node in self.graph}
        for deps in self.graph.values():
            for dep in deps:
                in_degree[dep] += 1

        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        self.sorted_nodes: list[HyperParameter] = []

        while queue:
            node = queue.popleft()
            self.sorted_nodes.append(node)
            for dep in self.graph[node]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        # If two nodes depend on each other, they can never be added to the
        # queue, hence we can compare the length to check for cycles:
        if len(self.sorted_nodes) != len(self.graph):
            raise ValueError("Cycle detected in hyperparameter dependencies!")

    def create_random_samples(self, n_samples: int) -> list[dict[str, Any]]:
        random_samples = []
        for _ in range(n_samples):
            random_sample = {}
            for node in self.sorted_nodes:
                if node.is_active(random_sample):
                    random_sample[node.name] = node.sample_parameter_random()
            random_samples.append(random_sample)
        return random_samples

    def create_grid_samples(self, n_samples: int) -> list[dict[str, Any]]:
        hp_grids = []
        for hp in self.sorted_nodes:
            hp_grids.append(hp.tuning_grid)
        total_param_grid = list(product(*hp_grids))
        # Resample the grid if the above division yielded to many points.
        # This of course will lead to some "holes" in the grid.
        if len(total_param_grid) > n_samples:
            total_param_grid = random.sample(total_param_grid, n_samples)
        elif len(total_param_grid) < n_samples:
            warnings.warn(
                f"""Defined tuning grids in given HyperParameters only yield 
                {len(total_param_grid)} combinations. To sample {n_samples}
                combinations, increase the 'default_grid' in the
                HyperParameters."""
            )
        # This are all possible combinations from HyperParameters,
        # but some may not be valid, since they are only active when
        # specific conditions are fulfilled, check this now.
        # TODO: What to do when HP output dependence on values of, another
        # HP?
        grid_samples = []
        for current_params in total_param_grid:
            config = {}
            for j, hp in enumerate(self.sorted_nodes):
                if hp.is_active(config):
                    config[hp.name] = current_params[j]
            grid_samples.append(config)
        return grid_samples

    # def _uniform_grid(self, n_samples: int) -> list[dict[str, Any]]:
    #     """Generates a *uniform* grid when all hyperparameters are independent
    #     of each other."""
    #     # First find out how many intervals and categorical parameters there are
    #     n_intervals = 0
    #     n_categorical = 0

    #     for param in self.sorted_nodes:
    #         if isinstance(param, CategoricalHyperparameter):
    #             n_categorical += len(param.parameter_range)
    #         else:
    #             n_intervals += 1
    #     # Divide total trials over all parameters (wanting to use all categorical ones)
    #     n_categorical = max(1, n_categorical)
    #     n_intervals = max(1, n_intervals)
    #     n_per_dim = int(math.ceil((n_samples / n_categorical) ** (1 / n_intervals)))

    #     # Create the point grid
    #     grid_axis = []
    #     for param in self.sorted_nodes:
    #         if isinstance(param, CategoricalHyperparameter):
    #             grid_axis.append(param.sample_parameter_grid(len(param.parameter_range)))
    #         else:
    #             grid_axis.append(param.sample_parameter_grid(n_per_dim))

    #     grid = list(product(*grid_axis))
    #     # Resample the grid if the above division yielded to many points.
    #     # This of course will lead to some "holes" in the grid.
    #     if len(grid) > n_samples:
    #         grid = [grid[i * len(grid) // n_samples] for i in range(n_samples)]

    #     grid_samples = []
    #     for i in range(n_samples):
    #         sample = {}
    #         counter = 0
    #         for node in self.sorted_nodes:
    #             sample[node.name] = grid[i][counter]
    #             counter += 1
    #         grid_samples.append(sample)

    #     return grid_samples
