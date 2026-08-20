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


def _leaf_names(variable: Variable) -> set[str]:
    return {leaf.name for leaf in variable.leaves}


def _prune(variable: Variable, keep_names: set[str]) -> Variable | None:
    """Removes every leaf of `variable` whose name isn't in `keep_names`,
    preserving as much of the original grouping as possible.

    Returns `variable` itself, unchanged, if none of its leaves needed
    dropping - so an auto-expanded multi-component variable used whole
    stays whole instead of being flattened into its individual components
    just because *some other, unrelated* variable needed pruning. Returns
    None if every leaf was dropped.
    """
    if variable.is_leaf:
        return variable if variable.name in keep_names else None
    survivors = [_prune(child, keep_names) for child in variable.children]
    survivors = [child for child in survivors if child is not None]
    if survivors == variable.children:
        return variable
    if not survivors:
        return None
    result = survivors[0]
    for child in survivors[1:]:
        result = result * child
    return result


def _segments(leaves: list[Variable], to_vars: list[Variable]) -> list[list[Variable]]:
    """The coarsest partition of `leaves` such that every `to_v`'s own leaf
    range, wherever it overlaps `leaves`, aligns exactly with a run of
    whole segments - so a variable nothing ever asks to subdivide comes
    back as a single segment (`leaves` itself, unsplit), and only the
    variables genuinely requested at finer granularity force a cut.

    A `to_v` doesn't have to be one contiguous run itself (e.g. `X * Z`
    when the source is `X * Y * Z` - X and Z aren't adjacent) - each
    maximal contiguous run *within* `to_v`'s own leaf order gets its own
    cut, so `X * Z` correctly forces `Y` into its own segment too (cut off
    from both sides) without requiring `X`/`Z` to be adjacent to each other.
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


def _compose(leaves: list[Variable]) -> Variable:
    if len(leaves) == 1:
        return leaves[0]
    result = leaves[0]
    for leaf in leaves[1:]:
        result = result * leaf
    return result


class PINNPipeline(Graph):
    """Automatically builds the computation graph for a PINN training pipeline
    based on the provided sampler, models, and constraint. The pipeline handles
    the necessary splitting and joining of variables, as well as tracking gradients
    for the computation of derivatives.

    Models (which can also be single Parameters) can not depend on outputs
    of another model, but are just executed separately.

    Args:
        sampler (DataNode): The data node that provides the input data for the PINN.
        models (list[Node]): A list of models that should be trained
        constraint (Constraint | None, optional): The constraint that describes
            the physics constrained that should be trained. If none
            is provided, a residual should be given instead. Defaults to None.
        residual (Callable | None, optional): The residual function to build
            a PINNConstraint from, if not constraint is provided.
            Defaults to None.
        residual_name (str | None, optional): A name for the constraint.
            Defaults to None.
        reduction (Callable | None, optional): A function that is coupled
            to the residual function to build the PINNConstraint. Defaults to None.
        weight (float, optional): A weight for the constraint. Defaults to 1.0.
        backend (_type_, optional): In which backend to build this graph in.
            Defaults to DEFAULT_DL_BACKEND.
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
            if residual_name is None:
                residual_name = "PINNConstraint"
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
        sampler_leaf_names = {name for sv in sampler_out_vars for name in _leaf_names(sv)}

        # Drop whatever leaves the sampler can't provide - pruning, not
        # flattening, so a constraint variable the sampler fully covers
        # stays exactly as composed (e.g. an auto-expanded 3D variable
        # stays whole instead of being split into its components).
        constrained_in_sampler_out = [
            pruned
            for v in constraint_input_vars
            if (pruned := _prune(v, sampler_leaf_names)) is not None
        ]

        # Model inputs the constraint doesn't already claim - also pruned
        # rather than flattened, and deduplicated by leaf name so two
        # models sharing an unclaimed variable don't add it twice.
        # here, still use original data configs since nothing was connected yet
        # -> nodes were not added to graph yet, so dynamic configs are not available yet
        only_model_input_vars: list[Variable] = []
        claimed_leaf_names = {
            name for cv in constraint_input_vars for name in _leaf_names(cv)
        }
        for model in models:
            for p in model.input_ports:
                mv = p.data_configuration.variables
                pruned = _prune(mv, _leaf_names(mv) - claimed_leaf_names)
                if pruned is not None:
                    only_model_input_vars.append(pruned)
                    claimed_leaf_names |= _leaf_names(pruned)

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
        if dynamic:
            return self.dynamic_data_configs[port.node][port].variables
        else:
            return port.data_configuration.variables

    def split_and_join(
        self, from_ports, to_vars, use_dynamic_data_configs=False
    ) -> list[OutputPort]:
        """Routes each port in `from_ports` to exactly the pieces each
        variable in `to_vars` needs.

        Splits only where some `to_v` genuinely requires a finer cut than
        what's already available - a `from_v` nothing ever asks to
        subdivide is passed straight through, no SplitVariables/
        ConcatVariables round-trip at all - and joins pieces back together
        only where a `to_v` spans more than one of them.
        """
        from_vars = [
            self.get_variables(p, dynamic=use_dynamic_data_configs) for p in from_ports
        ]

        # leaf name -> (port carrying it, the whole Variable piece that port outputs)
        leaf_source: dict[str, tuple] = {}
        for from_p, from_v in zip(from_ports, from_vars):
            leaves = from_v.leaves
            segments = _segments(leaves, to_vars)
            if len(segments) == 1:
                for leaf in leaves:
                    leaf_source[leaf.name] = (from_p, from_v)
                continue
            pieces = [_compose(segment) for segment in segments]
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
