from qewton.config.dtypes import Number

from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes
from qewton.algorithms.backend_node import BackendNode, TensorType


class ReLU(BackendNode[TensorType]):
    """Rectified Linear Unit (ReLU) activation function.

    Applies the ReLU activation function element-wise to the input tensor.
    Returns max(0, x) for each element, effectively zeroing out negative values.
    """

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


class Tanh(BackendNode[TensorType]):
    """Hyperbolic Tangent (Tanh) activation function.

    Applies the tanh activation function element-wise to the input tensor.
    Maps input values to the range [-1, 1], providing a smooth, differentiable activation.
    """

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


class Sigmoid(BackendNode[TensorType]):
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
