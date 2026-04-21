import torch
import pioneer
import pioneer.algorithms.building_blocks as bb

# lin = bb.Linear(5, 10)
# lin.setup()
# print(lin(input=torch.ones((3, 5))))
fcn = pioneer.algorithms.FCN(
    in_neurons=10,
    hidden_neurons=20,
    out_neurons=5,
    n_hidden_layers=1,
    activation=bb.ReLU,
)

X = 3.0 * torch.ones((100, 10))
print(fcn(input=X).shape)

X = 3.0 * torch.ones((3, 4))
add_layer = bb.math.Add()
print(add_layer(x=X, y=3.0))

mean_layer = bb.math.Mean()
print(mean_layer(x=X, axis=0))
