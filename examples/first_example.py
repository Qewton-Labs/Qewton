import numpy as np
import torch
import pioneer

x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + np.sin(6.0 * x_data)
data = torch.tensor(np.column_stack((x_data, u_data)), dtype=torch.float32)

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)
dataset = pioneer.data.DataSet.from_data(data, X * U, batch_size=1000)

slice_node = pioneer.nodes.SplitNode(dataset.data_config)

model = pioneer.algorithms.TorchFCN(X, U, 2, 8)

constraint = pioneer.constraints.MSEConstraint(
    model[model.OutputKeys.OUTPUT].data_configuration,
)

pipeline = pioneer.pipelines.Pipeline()

pipeline.connect(dataset, slice_node)
pipeline.connect(slice_node[X], model)
pipeline.connect(slice_node[U], constraint[constraint.InputKeys.INPUT1])
pipeline.connect(model, constraint[constraint.InputKeys.INPUT2])

pipeline.validate()
pipeline.visualize()
# runtime = pipeline.create_runtime()
# runtime.run()

trainer = pioneer.optim.trainer.PyTorchTrainer(
    [pipeline],
    training_constraints=[constraint],
    optimizer=torch.optim.Adam,
    max_iterations=5000,
    learning_rate=0.001,
    device="cpu",
)
trainer.run()
