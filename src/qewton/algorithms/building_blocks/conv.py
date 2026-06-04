# from qewton.base import LayerNode
# from qewton.implementation import TorchImplementation, DEFAULT_DL_BACKEND


# class TorchConv2d(TorchImplementation):
#     """Implementation of Conv2d layer in PyTorch backend."""

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         bias=True,
#         **kwargs,
#     ):
#         from torch.nn import Conv2d as TConv2d

#         super().__init__(
#             TConv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,
#                 groups=groups,
#                 bias=bias,
#                 **kwargs,
#             )
#         )


# class Conv2d(LayerNode):
#     """A node representing a 2D convolutional layer."""

#     existing_implementations = {TorchImplementation: TorchConv2d}

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         bias=True,
#         name="conv2d",
#         backend=DEFAULT_DL_BACKEND,
#         **kwargs,
#     ):
#         super().__init__(name=name, backend=backend, state=NodeState.FIXED)
#         self._input_ports[0].data_configuration.specify_dtype(backend)
#         self._output_ports[0].data_configuration.specify_dtype(backend)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.kwargs = kwargs
#         self.setup()

#     def setup(self):
#         self.implementation_instance = self.implementation(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#             groups=self.groups,
#             bias=self.bias,
#             **self.kwargs,
#         )
