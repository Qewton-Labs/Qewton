from qewton.config.dtypes import Number

from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes
from qewton.graphs.nodes import Node
from qewton.backends import TensorType


class ReLU(Node[TensorType]):
    """Rectified Linear Unit (ReLU) activation function.

    Applies the ReLU activation function element-wise to the input tensor.
    Returns max(0, x) for each element, effectively zeroing out negative values.
    """

    _type_identifier = "ReLUNode"
    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of ReLU activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with same shape and dtype as input.
        """
        return self.backend.nn.relu(x)


class Tanh(Node[TensorType]):
    """Hyperbolic Tangent (Tanh) activation function.

    Applies the tanh activation function element-wise to the input tensor.
    Maps input values to the range [-1, 1], providing a smooth, differentiable activation.
    """

    _type_identifier = "TanhNode"
    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of Tanh activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with values in range [-1, 1],
                same shape and dtype as input.
        """
        return self.backend.nn.tanh(x)


class Sigmoid(Node[TensorType]):
    """Sigmoid activation function.

    Applies the sigmoid activation function element-wise to the input tensor.
    Maps input values to the range (0, 1), providing a smooth probability-like output.

    Attributes:
        ellipsis_axes (EllipsisAxes): Configuration for tensor axes handling.

    Examples:
        >>> sigmoid = Sigmoid(backend)
        >>> x = backend.library.array([[-2, 0], [2, 4]])
        >>> output = sigmoid.forward(x)
        >>> # output: approximately [[0.12, 0.5], [0.88, 0.98]]
    """

    _type_identifier = "SigmoidNode"
    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of Sigmoid activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with values in range (0, 1),
                same shape and dtype as input.
        """
        return self.backend.nn.sigmoid(x)
