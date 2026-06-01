import pytest
import torch
from pioneer.config import Variable, DataConfiguration, BatchAxes, FeatureAxes, AxesDim
from pioneer.data import ArrayLikeDataSet, DataLoader
from pioneer.algorithms import FCN
from pioneer.constraints import PINNConstraint
from pioneer.graphs.pipelines import PINNPipeline
from pioneer.optim import OptimizationPhase, Adam, GraphBasedTrainer


@pytest.fixture
def simple_adam():
    return OptimizationPhase(
        optimizer=Adam(),
        lr=0.001,
        max_iterations=5,
    )


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
