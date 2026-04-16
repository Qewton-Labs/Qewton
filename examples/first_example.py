import torch
import pioneer

x_data = torch.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + torch.sin(6.0 * x_data)
data = torch.column_stack((x_data, u_data))

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)
dataset_X = pioneer.data.DataSet.from_data(x_data, X, batch_size=1000)
dataset_U = pioneer.data.DataSet.from_data(u_data, U, batch_size=1000)
# slice_node = pioneer.nodes.SplitNode(dataset.data_config)

model = pioneer.algorithms.FCN(
    in_neurons=1,
    hidden_neurons=50,
    out_neurons=1,
    n_hidden_layers=8,
    activation=pioneer.building_blocks.Tanh,
)

constraint = pioneer.constraints.MSEConstraint(
    model.output_ports[0].data_configuration,
)

pipeline = pioneer.Pipeline()

pipeline.connect(dataset_X, model)
pipeline.connect(model, constraint.input_1)
pipeline.connect(dataset_U, constraint.input_2)

# pipeline.setup()
# for connection in pipeline.sorted_edges:
#     for p, edge in connection.items():
#         print("Connected nodes:", edge.from_port.node.name, edge.to_port.node.name)
#         print("Connected ports:", edge.from_port.name, edge.to_port.name)
# pipeline.run()
# print(constraint.loss)
# params = pipeline.collect_trainable_parameters()
device = "cuda:0"
iterations = 50
trainer = pioneer.optim.trainer.PyTorchTrainer(
    [pipeline],
    training_constraints=[constraint],
    optimizer=torch.optim.Adam,
    max_iterations=iterations,
    learning_rate=0.001,
    device=device,
)

import time

trainer.run()

###################################################################
### Compare with default fcn and pytorch
import torch


activation = torch.nn.Tanh()
layers = []
layers.append(torch.nn.Linear(1, 50))
layers.append(activation)
for i in range(8):
    layers.append(torch.nn.Linear(50, 50))
    layers.append(activation)
layers.append(torch.nn.Linear(50, 1))

model_fcn = torch.nn.Sequential(*layers)

model_fcn.to(device)
x_data = x_data.to(device)
u_data = u_data.to(device)

optimizer = torch.optim.Adam(model_fcn.parameters(), lr=0.001)
start_time = time.time()
for i in range(iterations):
    start_time_eval = time.time()
    out = model_fcn(x_data)
    loss_value = torch.mean((out - u_data) ** 2)
    print("Computation step took:", time.time() - start_time_eval)

    start_time_loss = time.time()
    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Computation of Loss took:", time.time() - start_time_loss)
    if i % 100 == 0:
        print("Loss:", loss_value.item())
print("Training took:", time.time() - start_time)
