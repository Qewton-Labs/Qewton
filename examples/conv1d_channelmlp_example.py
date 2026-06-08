from qewton.algorithms.building_blocks.conv import FunctionalConv1d, Conv1d
from qewton.algorithms.dl_models.channel_mlp import ChannelMLP
from qewton.config.backend import TorchBackend, TensorflowBackend
import torch
import tensorflow as tf


c = FunctionalConv1d(backend=TensorflowBackend)
# x = torch.rand(10, 3, 7)
x = tf.random.uniform((10, 4, 7))
w = tf.random.uniform((2, 4, 3))
b = tf.random.uniform((2,))
print(c(x, w, b).shape)


c = FunctionalConv1d(backend=TorchBackend)
# x = torch.rand(10, 3, 7)
x = torch.rand((10, 4, 7))
w = torch.rand((2, 4, 3))
b = torch.rand((2,))
print(c(x, w, b).shape)

c = Conv1d(4, 2, 3, backend=TorchBackend)
print(c(x).shape)


c = ChannelMLP(2, backend=TorchBackend)
x = torch.rand(10, 2, 13)
print(c(x).shape)
