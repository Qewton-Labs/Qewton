from typing import Callable

from ...config.variables import Variable

from ...config.backend import DEFAULT_DL_BACKEND

from ...algorithms.building_blocks.derivatives import GradientTracking

from ...constraints.base import Constraint
from ...constraints.pinn_constraint import PINNConstraint
from ...data.dataloaders.base import DataLoader
from ..nodes import Node, Port
from ..graphs import Graph

## example functionality:

# def residual_fun(u: U, f: F, x: X):  # type: ignore
#     return u.gradient(x) - f


# constraint = pioneer.constraints.PINNConstraint(residual_fun, name="PINNConstraint")
# grad_tracking = pioneer.algorithms.building_blocks.GradientTracking()

# # pipeline = pioneer.graphs.PINNPipeline(constraint / residual_fun, model, sampler)

# computation_graph = pioneer.Graph()

# with computation_graph.tracker():
#     x, f = data_loader()
#     x = grad_tracking(x)
#     u = model(x)
#     constraint(u, f, x)


class PINNPipeline(Graph):
    """
    Models (which can also be single Parameters) can not depend on outputs
    of another model, but are just executed seperately.
    """
    
    def __init__(
        self,
        sampler: DataLoader,
        models: list[Node],
        constraint: Constraint | None = None,
        residual: Callable | None = None,
        reduction: Callable | None = None,
        weight=1.0,
        backend=DEFAULT_DL_BACKEND,
    ):
        super().__init__()

        if constraint is None:
            assert residual is not None, "Either constraint or residual must be provided."
            constraint = PINNConstraint(
                residual, reduction, weight=weight, backend=backend
            )
        
        split_ports = {}
        for sampler_port in sampler.output_ports:
            if len(sampler_port.data_configuration.variables) > 1:
                split = SplitVariables()
                self.connect(sampler_port, split)
                for var in sampler_port.data_configuration.variables:
                    split_ports[var] = split.get_output_port(var)
            else:
                split_ports[sampler_port.data_configuration.variables.keys()[0]] = sampler_port
        
        tracked_vars = []
        for model in models:
            for model_in_port in model.input_ports:
                vars = model_in_port.data_configuration.variables
                if len(vars) > 1:
                    join = ConcatVariables(vars)
                for var in vars:
                    if var not in tracked_vars:
                        tracker = GradientTracking()
                        self.connect(split_ports[var], tracker)
                        split_ports[var] = tracker.output_ports[0]
                        tracked_vars.append(var)
                    if len(vars) > 1:
                        self.connect(tracker, join.get_input_port(var))
                if len(vars) > 1:
                    self.connect(join, model_in_port)
                else:
                    self.connect(split_ports[var], model_in_port)
            

        

    def _find_common_var_splitting(
        self, from_ports, to_ports_a, to_ports_b
    ) -> list[SplitVariablesNode], dict[Port, Port], dict[Port, (Port, Port)]:

        return
