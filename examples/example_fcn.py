import torch
import pioneer
import pioneer.algorithms.building_blocks as bb

fcn = pioneer.algorithms.FCN(
    in_neurons=10,
    hidden_neurons=20,
    out_neurons=5,
    n_hidden_layers=2,
    bias=True,
    activation=pioneer.building_blocks.ReLU,
)

X = 3.0 * torch.ones((10, 10))
print(fcn(input=X))


add_layer = bb.math.Add()
print(add_layer(input1=X, input2=3.0))


mean_layer = bb.math.Mean(axis=0)
print(mean_layer(input=X))
