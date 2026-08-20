import pytest
import torch
from qewton.config import Variable, DataConfiguration, BatchAxes, FeatureAxes, AxesDim
from qewton.data import ArrayLikeDataSet, DataLoader
from qewton.algorithms import FCN
from qewton.constraints import PINNConstraint
from qewton.graphs.control_nodes.wrapper_node import FunctionWrappingNode
from qewton.graphs.pipelines import PINNPipeline
from qewton.graphs.pipelines.pinn_pipeline import _compose, _prune, _segments
from qewton.optim import OptimizationPhase, Adam, GraphBasedTrainer


@pytest.fixture
def simple_adam():
    return OptimizationPhase(
        optimizer=Adam(),
        lr=0.001,
        max_iterations=5,
    )


class TestPrune:
    def test_returns_the_same_object_when_nothing_is_dropped(self):
        """The whole point: a variable nothing needs to cut stays whole -
        confirmed by identity, not just structural equality."""
        pos = Variable("pos", 3)
        assert _prune(pos, {"pos_0", "pos_1", "pos_2"}) is pos

    def test_drops_a_leaf_not_in_keep_names(self):
        x, y = Variable("x", 1), Variable("y", 1)
        xy = x * y
        pruned = _prune(xy, {"x"})
        assert pruned is x

    def test_returns_none_when_everything_is_dropped(self):
        x, y = Variable("x", 1), Variable("y", 1)
        assert _prune(x * y, set()) is None

    def test_a_leaf_survives_or_not_on_its_own(self):
        x = Variable("x", 1)
        assert _prune(x, {"x"}) is x
        assert _prune(x, set()) is None


class TestSegments:
    def test_a_variable_nothing_subdivides_stays_one_segment(self):
        x, y, z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        leaves = [x, y, z]
        segments = _segments(leaves, [x * y * z])
        assert segments == [leaves]

    def test_a_requested_sub_leaf_forces_a_cut(self):
        x, y, z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        leaves = [x, y, z]
        segments = _segments(leaves, [x * y * z, y])
        assert segments == [[x], [y], [z]]

    def test_a_non_adjacent_request_cuts_out_whats_in_between_too(self):
        """X * Z (skipping Y) forces Y into its own segment too, even
        though nothing directly asked for Y alone."""
        x, y, z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        leaves = [x, y, z]
        segments = _segments(leaves, [x * z])
        assert segments == [[x], [y], [z]]

    def test_unrelated_leaves_do_not_force_a_cut(self):
        x, y = Variable("x", 1), Variable("y", 1)
        f = Variable("f", 1)
        leaves = [x, y]
        segments = _segments(leaves, [x * y, f])  # f isn't part of these leaves at all
        assert segments == [leaves]


class TestCompose:
    def test_a_single_leaf_is_returned_unchanged(self):
        x = Variable("x", 1)
        assert _compose([x]) is x

    def test_multiple_leaves_are_composed_in_order(self):
        x, y = Variable("x", 1), Variable("y", 1)
        composed = _compose([x, y])
        assert composed.leaves == [x, y]
        assert composed.dim == 2


def test_pinn_pipeline_basic_execution(simple_adam):
    """Tests a standard 1D PINN pipeline setup."""
    x_data = torch.linspace(0, 1, 10).reshape(-1, 1)
    f_data = 2.0 * x_data

    X = Variable("x", 1)
    U = Variable("u", 1)
    F = Variable("f", 1)

    x_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(X))
    f_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(F))

    dataset = ArrayLikeDataSet(data=[x_data, f_data], data_configs=[x_config, f_config])
    data_loader = DataLoader(dataset, batch_size=10)

    model = FCN(in_neurons=X, hidden_neurons=5, out_neurons=U, n_hidden_layers=1)

    def residual_fun(u: U, f: F, x: X):  # type: ignore
        return u.gradient(x) - f

    constraint = PINNConstraint(residual_fun)
    pipeline = PINNPipeline(data_loader, [model], constraint)
    pipeline.setup()

    trainer = GraphBasedTrainer(
        optimization_phases=[simple_adam],
        graphs=[pipeline],
        training_objectives=[constraint],
        device="cpu",
    )
    trainer.run()
    assert trainer.train_state.iteration == 5


def test_pinn_pipeline_track_residual_false_allows_raw_backend_ops(simple_adam):
    """track_residual=False builds the residual as a FunctionWrappingNode,
    which executes eagerly on real tensors instead of being traced through
    TrackingObject - so a residual using a raw torch.autograd call (not
    `u.gradient(x)`, which only TrackingObject understands how to trace)
    still works. PINNPipeline's own gradient-tracking pass is unaffected -
    it acts on ports/variables, not on how the residual itself is wrapped -
    so `x` already has requires_grad set by the time it reaches here."""
    x_data = torch.linspace(0, 1, 10).reshape(-1, 1)
    f_data = 2.0 * x_data

    X = Variable("x", 1)
    U = Variable("u", 1)
    F = Variable("f", 1)

    x_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(X))
    f_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(F))

    dataset = ArrayLikeDataSet(data=[x_data, f_data], data_configs=[x_config, f_config])
    data_loader = DataLoader(dataset, batch_size=10)

    model = FCN(in_neurons=X, hidden_neurons=5, out_neurons=U, n_hidden_layers=1)

    def residual_fun(u: U, f: F, x: X):  # type: ignore
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x - f

    # residual= + track_residual=, not a pre-built constraint - exercises
    # PINNPipeline's own forwarding of track_residual into the PINNConstraint
    # it constructs internally, not just PINNConstraint in isolation.
    pipeline = PINNPipeline(
        data_loader, [model], residual=residual_fun, track_residual=False
    )
    assert isinstance(pipeline.constraint.residual_node, FunctionWrappingNode)
    constraint = pipeline.constraint
    pipeline.setup()

    trainer = GraphBasedTrainer(
        optimization_phases=[simple_adam],
        graphs=[pipeline],
        training_objectives=[constraint],
        device="cpu",
    )
    trainer.run()
    assert trainer.train_state.iteration == 5


def test_pinn_pipeline_with_input_splitting(simple_adam):
    """
    Tests a case where a concatenated input XY from the sampler must be
    split into X and Y to calculate individual gradients in the constraint.
    """
    X = Variable("x", 1)
    Y = Variable("y", 1)
    U = Variable("u", 1)
    F = Variable("f", 1)
    XY = X * Y

    xy_data = torch.rand((20, 2))
    f_data = torch.rand((20, 1))

    xy_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(XY))
    f_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(F))

    dataset = ArrayLikeDataSet([xy_data, f_data], [xy_config, f_config])
    data_loader = DataLoader(dataset, batch_size=10)

    # Model takes the combined XY input
    model = FCN(in_neurons=XY, hidden_neurons=10, out_neurons=U, n_hidden_layers=1)

    # Residual requires gradients w.r.t X and Y separately
    def residual_fun(u: U, x: X, y: Y, f: F):  # type: ignore
        return u.gradient(x) + u.gradient(y) - f

    constraint = PINNConstraint(residual_fun)
    pipeline = PINNPipeline(data_loader, [model], constraint)
    pipeline.setup()

    trainer = GraphBasedTrainer(
        optimization_phases=[simple_adam],
        graphs=[pipeline],
        training_objectives=[constraint],
        device="cpu",
    )
    trainer.run()
    assert trainer.train_state.iteration == 5


def test_pinn_pipeline_multi_model_concatenation(simple_adam):
    """
    Tests multiple models whose outputs are joined into a single
    variable for the constraint residual.
    """
    X = Variable("x", 1)
    Y = Variable("y", 1)
    U = Variable("u", 1)
    V = Variable("v", 1)
    F = Variable("f", 1)
    UV = U * V  # Target concatenated variable

    x_data = torch.rand((15, 1))
    y_data = torch.rand((15, 1))
    f_data = torch.rand((15, 1))

    dataset = ArrayLikeDataSet(
        [x_data, y_data, f_data],
        [
            DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(X)),
            DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(Y)),
            DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(F)),
        ],
    )
    data_loader = DataLoader(dataset, batch_size=5)

    model_u = FCN(in_neurons=X, out_neurons=U, hidden_neurons=5, n_hidden_layers=1)
    model_v = FCN(in_neurons=Y, out_neurons=V, hidden_neurons=5, n_hidden_layers=1)

    def residual_fun(uv: UV, f: F):  # type: ignore
        # Slice node logic: extracting variables from a concatenated input
        return uv[U] + uv[V] - f

    constraint = PINNConstraint(residual_fun)
    pipeline = PINNPipeline(data_loader, [model_u, model_v], constraint)
    pipeline.setup()

    trainer = GraphBasedTrainer(
        optimization_phases=[simple_adam],
        graphs=[pipeline],
        training_objectives=[constraint],
        device="cpu",
    )
    trainer.run()
    assert trainer.train_state.iteration == 5


def test_pinn_pipeline_mixed_split_concat(simple_adam):
    """
    Tests a mixed case where:
    1. Sampler provides XYZ and F.
    2. Model takes only Y (requires split from XYZ).
    3. Constraint takes XZ (requires concat of X and Z) and U (model output).
    """
    X = Variable("x", 1)
    Y = Variable("y", 1)
    Z = Variable("z", 1)
    U = Variable("u", 1)
    F = Variable("f", 1)
    XYZ = X * Y * Z
    XZ = X * Z

    xyz_data = torch.rand((20, 3))
    f_data = torch.rand((20, 1))

    xyz_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(XYZ))
    f_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(F))

    dataset = ArrayLikeDataSet([xyz_data, f_data], [xyz_config, f_config])
    data_loader = DataLoader(dataset, batch_size=10)

    # Model takes only Y
    model = FCN(in_neurons=Y, hidden_neurons=10, out_neurons=U, n_hidden_layers=1)

    # Residual requires combined XZ, model output U, and F
    def residual_fun(xz: XZ, u: U, f: F):  # type: ignore
        # Accessing X and Z via slicing of the concatenated XZ
        return xz[X] + xz[Z] + u - f

    constraint = PINNConstraint(residual_fun)
    pipeline = PINNPipeline(data_loader, [model], constraint)
    pipeline.setup()

    trainer = GraphBasedTrainer(
        optimization_phases=[simple_adam],
        graphs=[pipeline],
        training_objectives=[constraint],
        device="cpu",
    )
    trainer.run()
    assert trainer.train_state.iteration == 5
