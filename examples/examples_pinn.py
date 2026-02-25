import numpy as np
import torch
import pioneer

x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2
data_ode = torch.tensor(np.column_stack((x_data, u_data)), dtype=torch.float32)
data_initial = torch.tensor(x_data[:1], dtype=torch.float32)

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)

dataset_ode = pioneer.data.DataSet.from_data(data_ode, X * U, batch_size=1000)
dataset_initial = pioneer.data.DataSet.from_data(data_initial, X, batch_size=1000)

model = pioneer.algorithms.TorchFCN(X, U, 2, 8)


### Constraints for training (with PINNs)
# ODE:
def ode_residual(x, u):
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return u_x - 2 * x


slice_node = pioneer.nodes.SplitNode(dataset_ode.data_config)

tracking_node = pioneer.nodes.GradientTrackingNode(
    model[model.InputKeys.INPUT].data_configuration
)

ode_constraint = pioneer.constraints.ResidualConstraint(
    model[model.InputKeys.INPUT].data_configuration,
    model[model.OutputKeys.OUTPUT].data_configuration,
    residual_fn=ode_residual,
    name="ODE_Constraint",
)

ode_pipeline = pioneer.pipelines.Pipeline(name="ode_pipeline")

ode_pipeline.connect(
    dataset_ode[dataset_ode.OutputKeys.OUTPUT], slice_node[dataset_ode.InputKeys.INPUT]
)

ode_pipeline.connect(
    slice_node[X],
    tracking_node[tracking_node.InputKeys.INPUT],
)
ode_pipeline.connect(
    tracking_node[tracking_node.OutputKeys.OUTPUT], model[model.InputKeys.INPUT]
)
ode_pipeline.connect(
    tracking_node[tracking_node.OutputKeys.OUTPUT],
    ode_constraint[ode_constraint.InputKeys.INPUT1],
)
ode_pipeline.connect(
    model[model.OutputKeys.OUTPUT], ode_constraint[ode_constraint.InputKeys.INPUT2]
)

# Testing can go into the same pipeline
test_constraint = pioneer.constraints.MSEConstraint(
    model[model.OutputKeys.OUTPUT].data_configuration, name="Validation Constraint"
)

ode_pipeline.connect(
    slice_node[U],
    test_constraint[test_constraint.InputKeys.INPUT1],
)
ode_pipeline.connect(
    model[model.OutputKeys.OUTPUT],
    test_constraint[test_constraint.InputKeys.INPUT2],
)


ode_pipeline.validate()
ode_pipeline.visualize()


# Initial condition
def initial_residual(_x, u):
    return u


initial_constraint = pioneer.constraints.ResidualConstraint(
    model[model.InputKeys.INPUT].data_configuration,
    model[model.OutputKeys.OUTPUT].data_configuration,
    residual_fn=initial_residual,
    name="Initial",
)

initial_pipeline = pioneer.pipelines.Pipeline(name="initial_pipeline")

initial_pipeline.connect(
    dataset_initial[dataset_ode.OutputKeys.OUTPUT],
    model[model.InputKeys.INPUT],
)

initial_pipeline.connect(
    dataset_initial[dataset_ode.OutputKeys.OUTPUT],
    initial_constraint[initial_constraint.InputKeys.INPUT1],
)
initial_pipeline.connect(
    model[model.OutputKeys.OUTPUT],
    initial_constraint[initial_constraint.InputKeys.INPUT2],
)

initial_pipeline.validate()

# Start training:
trainer = pioneer.optim.trainer.PyTorchTrainer(
    [ode_pipeline, initial_pipeline],
    training_constraints=[initial_constraint, ode_constraint],
    optimizer=torch.optim.Adam,
    max_iterations=5000,
    learning_rate=0.001,
    device="cpu",
    validation_constraints=[test_constraint],
)
trainer.run()
