import torch
import pioneer

fcn = pioneer.algorithms.FCN(
    in_neurons=10,
    hidden_neurons=20,
    out_neurons=5,
    n_hidden_layers=2,
    bias=True,
    activation=pioneer.building_blocks.ReLU,
)
