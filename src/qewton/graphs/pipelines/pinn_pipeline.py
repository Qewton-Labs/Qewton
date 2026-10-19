"""Graph template wiring a sampler, models and a physics constraint into
a PINN training pipeline."""

from typing import Callable

from qewton.algorithms.building_blocks.array_operations import (
    SplitVariables,
    ConcatVariables,
)

from qewton.config.variables import Variable

from qewton.backends import DEFAULT_DL_BACKEND

from qewton.algorithms.building_blocks.derivatives import GradientTracking

from qewton.constraints.base import Constraint
from qewton.constraints.pinn_constraint import PINNConstraint
from qewton.data.dataloaders.base import DataNode
from qewton.graphs.nodes import Node, OutputPort
from qewton.graphs import Graph


def _variable_segments(leaves: list[Variable], to_vars: list[Variable]) -> list[list[Variable]]:
    """The coarsest partition of `leaves` such that the leaves of every
    variable in `to_vars` that lie in `leaves` form whole segments.

    Without any overlapping `to_vars`, `leaves` is returned as a single
    segment. A variable whose leaves are not contiguous in `leaves` (e.g.
    `X * Z` for `X * Y * Z`) cuts around each contiguous run, so `Y` ends
    up in its own segment.
    """
    index = {leaf.name: i for i, leaf in enumerate(leaves)}
    cuts = {0, len(leaves)}
    for to_v in to_vars:
        positions = [index[leaf.name] for leaf in to_v.leaves if leaf.name in index]
        if not positions:
            continue
        run_start = prev = positions[0]
        for pos in positions[1:]:
            if pos != prev + 1:
                cuts.add(run_start)
                cuts.add(prev + 1)
                run_start = pos
            prev = pos
        cuts.add(run_start)
        cuts.add(prev + 1)
    boundaries = sorted(cuts)
    return [leaves[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]


class PINNPipeline(Graph):
    """Graph connecting a sampler, models and a constraint for PINN training.

    The output variables of `sampler` are split and joined as needed by
    the models and the constraint, and every sampled variable a model
    uses passes through a
    :class:`~qewton.algorithms.building_blocks.derivatives.GradientTracking`
    node, so the constraint can differentiate the model outputs with
    respect to it. Constraint inputs are taken from the sampler outputs
    and the model outputs. Variables are matched by leaf name. A variable
    is only split where a consumer needs a finer part of it.

    Models (which can also be single parameters) are evaluated
    independently; a model cannot depend on the output of another model.

    Args:
        sampler: The data node providing the input points.
        models: The models to train.
        constraint: The physics constraint to train. None (default) builds a
            :class:`~qewton.constraints.pinn_constraint.PINNConstraint` from
            `residual`.
        residual: The residual function of the constraint built when
            `constraint` is None.
        residual_name: The name of the built constraint. None (default) uses
            the name of the residual function.
        track_residual: Passed to the built constraint.
        reduction: The reduction of the built constraint.
        weight (float, optional): The weight of the built constraint. Default
            is 1.0.
        backend (type[Backend], optional): The backend of the built constraint.
            Default is ``DEFAULT_DL_BACKEND``.

    Raises:
        AssertionError: If neither `constraint` nor `residual` is given.
        ValueError: If a variable needed by a model or the constraint is not
            provided by the sampler or the models.

    Attributes:
        constraint (Constraint): The given or built constraint.
    """

    def __init__(
        self,
        sampler: DataNode,
        models: list[Node],
        constraint: Constraint | None = None,
        residual: Callable | None = None,
        residual_name: str | None = None,
        track_residual: bool = True,
        reduction: Callable | None = None,
        weight=1.0,
        backend=DEFAULT_DL_BACKEND,
    ):
        super().__init__()

        if constraint is None:
            assert residual is not None, "Either constraint or residual must be provided."
            self.constraint = PINNConstraint(
                residual,
                reduction,
                weight=weight,
                track_residual=track_residual,
                backend=backend,
                name=residual_name,
            )
        else:
            self.constraint = constraint

        # first: split and track
        constraint_input_vars = [
            p.data_configuration.variables for p in self.constraint.input_ports
        ]
        sampler_out_vars = [p.data_configuration.variables for p in sampler.output_ports]
        sampler_leaf_names = {name for sv in sampler_out_vars for name in sv.leaf_names}

        # Drop the leaves the sampler cannot provide. Pruning keeps a fully
        # covered constraint variable exactly as composed.
        constrained_in_sampler_out = [
            pruned
            for v in constraint_input_vars
            if (pruned := v.prune(sampler_leaf_names)) is not None
        ]

        # Model inputs not already claimed by the constraint, pruned and
        # deduplicated by leaf name. Static configs are used since no node
        # is part of the graph yet.
        only_model_input_vars: list[Variable] = []
        claimed_leaf_names = {
            name for cv in constraint_input_vars for name in cv.leaf_names
        }
        for model in models:
            for p in model.input_ports:
                mv = p.data_configuration.variables
                pruned = mv.prune(mv.leaf_names - claimed_leaf_names)
                if pruned is not None:
                    only_model_input_vars.append(pruned)
                    claimed_leaf_names |= pruned.leaf_names

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
                        leaf in p.data_configuration.variables
                        for leaf in model_p.data_configuration.variables.leaves
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
        for p_out, p_in in zip(out_ports, self.constraint.input_ports):
            self.connect(p_out, p_in)

    def get_variables(self, port, dynamic=False):
        """Return the variables of the data configuration of `port`.

        Args:
            port (Port): The port to read.
            dynamic (bool, optional): Whether to read the configuration of
                `port` in this graph instead of its static one. Default is
                False.

        Returns:
            Variable: The (composed) variable of the configuration.
        """
        if dynamic:
            return self.dynamic_data_configs[port.node][port].variables
        else:
            return port.data_configuration.variables

    def split_and_join(
        self, from_ports, to_vars, use_dynamic_data_configs=False
    ) -> list[OutputPort]:
        """Provide one output port per variable in `to_vars` from
        `from_ports`.

        A port of `from_ports` is only split (with
        :class:`~qewton.algorithms.building_blocks.array_operations.SplitVariables`)
        where some variable of `to_vars` needs a finer part of it;
        otherwise it is used directly. Pieces are joined (with
        :class:`~qewton.algorithms.building_blocks.array_operations.ConcatVariables`)
        only where a variable of `to_vars` spans more than one of them. The
        created nodes are connected in this graph.

        Args:
            from_ports (list[OutputPort]): The ports providing the data.
            to_vars (list[Variable]): The variables to provide, matched to
                `from_ports` by leaf name.
            use_dynamic_data_configs (bool, optional): Whether to read the
                variables of `from_ports` from their configurations in this
                graph instead of the static ones. Default is False.

        Returns:
            list[OutputPort]: For each variable in `to_vars`, the port
                providing it.

        Raises:
            ValueError: If a variable of `to_vars` needs a leaf not provided by
                `from_ports`.
        """
        from_vars = [
            self.get_variables(p, dynamic=use_dynamic_data_configs) for p in from_ports
        ]

        # leaf name -> (port carrying it, the whole variable that port outputs)
        leaf_source: dict[str, tuple] = {}
        for from_p, from_v in zip(from_ports, from_vars):
            leaves = from_v.leaves
            segments = _variable_segments(leaves, to_vars)
            if len(segments) == 1:
                for leaf in leaves:
                    leaf_source[leaf.name] = (from_p, from_v)
                continue
            pieces = [Variable.compose(segment) for segment in segments]
            split = SplitVariables(pieces)
            self.connect(from_p, split)
            for segment, piece in zip(segments, pieces):
                port = split.get_output_port(piece)
                for leaf in segment:
                    leaf_source[leaf.name] = (port, piece)

        out_ports = []
        for to_v in to_vars:
            leaves = to_v.leaves
            try:
                sources = [leaf_source[leaf.name] for leaf in leaves]
            except KeyError as exc:
                raise ValueError(
                    f"{to_v.name} needs a variable not provided by any of the "
                    "given from_ports."
                ) from exc

            pieces_needed = []
            for source in sources:
                if not pieces_needed or pieces_needed[-1] != source:
                    pieces_needed.append(source)

            if len(pieces_needed) == 1 and pieces_needed[0][1].leaves == leaves:
                out_ports.append(pieces_needed[0][0])
            else:
                join_vars = [piece for _, piece in pieces_needed]
                join = ConcatVariables(join_vars)
                for port, piece in pieces_needed:
                    self.connect(port, join.get_input_port(piece.name))
                out_ports.append(join.output_ports[0])

        return out_ports
