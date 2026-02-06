import numpy as np
import torch
import pioneer


x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + np.sin(6.0 * x_data)
data = torch.tensor(np.column_stack((x_data, u_data)), dtype=torch.float32)

X = pioneer.configurations.Variable("x", 1)
U = pioneer.configurations.Variable("u", 1)
dataset = pioneer.nodes.DataSet.from_data(data, X * U, batch_size=1000)

slice_node = pioneer.nodes.SliceNode(dataset.data_config)

model = pioneer.algorithms.TorchFCN(X, U, 2, 8)

constrain = pioneer.constraints.MSEConstraint(
    model[model.OutputKeys.OUTPUT].data_configuration,
    pioneer.optimization.EvaluationMode.TRAIN,
)

pipeline = pioneer.pipeline.Pipeline()

pipeline.connect(dataset[dataset.OutputKeys.OUTPUT], slice_node[dataset.InputKeys.INPUT])
pipeline.connect(slice_node["x"], model[dataset.InputKeys.INPUT])
pipeline.connect(slice_node["u"], constrain[constrain.InputKeys.INPUT1])
pipeline.connect(model[model.OutputKeys.OUTPUT], constrain[constrain.InputKeys.INPUT2])

pipeline.validate()
# runtime = pipeline.create_runtime()
# runtime.run()

optimizer = pioneer.optimization.trainer.TorchBackend(torch.optim.Adam, lr=0.001)
trainer = pioneer.optimization.trainer.Trainer(
    optimizer, [pipeline], max_iterations=5000, device="cpu"
)
trainer.run()
