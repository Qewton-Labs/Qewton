import numpy as np
import torch
import pioneer

x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + np.sin(6.0 * x_data)
data = torch.tensor(np.column_stack((x_data, u_data)), dtype=torch.float32)

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)
dataset = pioneer.nodes.DataSet.from_data(data, X * U, batch_size=1000)

slice_node = pioneer.nodes.SliceNode(dataset.data_config)

model = pioneer.algorithms.TorchFCN(
    X,
    U,
    hidden_layers=pioneer.optim.DiscreteHyperparameter((1, 3)),
    hidden_neurons=pioneer.optim.DiscreteHyperparameter((1, 16)),
    activation_fn=pioneer.optim.CategoricalHyperparameter(
        [torch.nn.Tanh(), torch.nn.ReLU()]
    ),
)

constrain = pioneer.constraints.MSEConstraint(
    model[model.OutputKeys.OUTPUT].data_configuration,
)

pipeline = pioneer.pipeline.Pipeline()

pipeline.connect(dataset[dataset.OutputKeys.OUTPUT], slice_node[dataset.InputKeys.INPUT])
pipeline.connect(slice_node[X], model[dataset.InputKeys.INPUT])
pipeline.connect(slice_node[U], constrain[constrain.InputKeys.INPUT1])
pipeline.connect(model[model.OutputKeys.OUTPUT], constrain[constrain.InputKeys.INPUT2])

pipeline.validate()

trainer = pioneer.optim.trainer.PyTorchTrainer(
    [pipeline],
    [constrain],
    torch.optim.Adam,
    max_iterations=pioneer.optim.CategoricalHyperparameter([1000, 2000, 5000]),
    learning_rate=0.001,
    device="cpu",
)

tuner = pioneer.optim.tuner.RandomSearchTuner(trainer, [constrain], 40, 4)
tuner.run()
