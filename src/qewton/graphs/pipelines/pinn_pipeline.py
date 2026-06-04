from typing import Callable

from qewton.algorithms.building_blocks.array_operations import SplitVariables, ConcatVariables

from qewton.config.variables import Variable

from qewton.config.backend import DEFAULT_DL_BACKEND

from qewton.algorithms.building_blocks.derivatives import GradientTracking

from qewton.constraints.base import Constraint
from qewton.constraints.pinn_constraint import PINNConstraint
from qewton.data.dataloaders.base import DataNode
from qewton.graphs.nodes import Node, OutputPort
from qewton.graphs import Graph


class PINNPipeline(Graph):
    """
    Models (which can also be single Parameters) can not depend on outputs
    of another model, but are just executed seperately.
    """

    def __init__(
        self,
        sampler: DataNode,
        models: list[Node],
        constraint: Constraint | None = None,
        residual: Callable | None = None,
        residual_name: str | None = None,
        reduction: Callable | None = None,
        weight=1.0,
        backend=DEFAULT_DL_BACKEND,
    ):
        super().__init__()

        if constraint is None:
            assert residual is not None, "Either constraint or residual must be provided."
            if residual_name is None:
                residual_name = "PINNConstraint"
            constraint = PINNConstraint(
                residual, reduction, weight=weight, backend=backend, name=residual_name
            )

        # first: split and track
        constraint_input_vars = [
            p.data_configuration.variables for p in constraint.input_ports
        ]
        sampler_out_vars = [p.data_configuration.variables for p in sampler.output_ports]

        # TODO: reduce to the stuff which is provided by the sampler
        # if a variable is not in the sampler but in

        # Filter constraint variables: remove keys not in sampler.
        # If no keys remain for a variable, it is omitted entirely.
        constrained_in_sampler_out = [
            Variable.from_dict(
                {k: d for k, d in v.items() if any(k in sv for sv in sampler_out_vars)}
            )
            for v in constraint_input_vars
        ]
        constrained_in_sampler_out = [
            v for v in constrained_in_sampler_out if not v.is_empty()
        ]
        # here, still use original data configs since nothing was connected yet
        # -> nodes were not added to graph yet, so dynamic configs are not available yet
        only_model_input_vars = []
        for model in models:
            for p in model.input_ports:
                for k, dim in p.data_configuration.variables.items():
                    if not any(k in cv for cv in constraint_input_vars):
                        if not any(k in mv for mv in only_model_input_vars):
                            only_model_input_vars.append(Variable(k, dim))

        trackable_ports = self.split_and_join(
            sampler.output_ports,
            constrained_in_sampler_out + only_model_input_vars,
            use_dynamic_data_configs=False,
        )

        for i, p in enumerate(trackable_ports):
            for model in models:
                found = False
                for model_p in model.input_ports:
                    if any(
                        [
                            k in p.data_configuration.variables
                            for k in model_p.data_configuration.variables
                        ]
                    ):
                        tracking = GradientTracking()
                        self.connect(p, tracking)
                        trackable_ports[i] = tracking.output_ports[0]
                        found = True
                        break
                if found:
                    break

        # step 2: connect these ports to models where necessary
        for model in models:
            model_in_vars = [p.data_configuration.variables for p in model.input_ports]
            model_in_ports = self.split_and_join(
                trackable_ports, model_in_vars, use_dynamic_data_configs=True
            )
            for p_in, model_p in zip(model_in_ports, model.input_ports):
                self.connect(p_in, model_p)

        # step 3: append model outputs
        for model in models:
            trackable_ports.extend(model.output_ports)

        # step 4: connect to constraint
        # TODO: test whether multi-key variables work correctly in constraints
        out_ports = self.split_and_join(
            trackable_ports, constraint_input_vars, use_dynamic_data_configs=True
        )
        for p_out, p_in in zip(out_ports, constraint.input_ports):
            self.connect(p_out, p_in)

    def get_variables(self, port, dynamic=False):
        if dynamic:
            return self.dynamic_data_configs[port.node][port].variables
        else:
            return port.data_configuration.variables

    def split_and_join(
        self, from_ports, to_vars, use_dynamic_data_configs=False
    ) -> list[OutputPort]:
        from_vars = [
            self.get_variables(p, dynamic=use_dynamic_data_configs) for p in from_ports
        ]
        out_ports = []
        split_ports = {}
        for from_p, from_v in zip(from_ports, from_vars):
            if len(from_v) > 1:
                split = SplitVariables()
                self.connect(from_p, split)
                for var in from_v:
                    split_ports[var] = split.get_output_port(var)
            else:
                split_ports[list(from_v.keys())[0]] = from_p

        for to_v in to_vars:
            if len(to_v) > 1:
                join = ConcatVariables(
                    [Variable(name, dim) for name, dim in to_v.items()]
                )
                for var in to_v:
                    self.connect(split_ports[var], join.get_input_port(var))
                out_ports.append(join.output_ports[0])
            else:
                var_name = next(iter(to_v))
                out_ports.append(split_ports[var_name])

        return out_ports
