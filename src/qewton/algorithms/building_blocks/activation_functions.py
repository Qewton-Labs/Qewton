from qewton.config.dtypes import Number

from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes
from qewton.graphs.nodes import Node
from qewton.backends import TensorType
from qewton.algorithms.building_blocks.utility import Cast
from qewton.algorithms.building_blocks.complex import ReIm
from qewton.backends import Backend, DEFAULT_DL_BACKEND
from qewton.graphs.control_nodes.graph_node import TrackedNode, CopiedNode


class ReLU(Node[TensorType]):
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


class Tanh(Node[TensorType]):
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

    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)]
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of Sigmoid activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with values in range [0, 1],
                same shape and dtype as input.
        """
        return self.backend.library.sigmoid(x)


class GELU(Node[TensorType]):
    """Gaussian Error Linear Unit (GELU) activation function.

    Applies the GELU activation function element-wise to the input tensor.
    Weights input values according to the cumulative distribution function
    of a standard Gaussian distribution, providing a smooth nonlinear output.

    Attributes:
        ellipsis_axes (EllipsisAxes): Configuration for tensor axes handling.

    Examples:
        >>> gelu = GELU(backend)
        >>> x = backend.library.array([[-2, 0], [2, 4]])
        >>> output = gelu.forward(x)
        >>> # output: approximately [[-0.05, 0], [1.95, 4.0]]
    """

    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of GELU activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with values in range [0, inf],
                same shape and dtype as input.
        """
        return self.backend.library.gelu(x)


class CGELU(TrackedNode):
    """"Complex Gaussian Error Linear Unit (CGELU) activation function.

    Applies the GELU activation function independently to the real and
    imaginary parts of a complex-valued input tensor, then recombines
    the activated components into a complex-valued output tensor.

    Attributes:
        ellipsis_axes (EllipsisAxes): Configuration for tensor axes handling.

    Examples:
        >>> cgelu = CGELU(backend)
        >>> x = backend.library.array([[-2 + 1j, 0 + 2j], [2 - 1j, 4 + 3j]])
        >>> output = cgelu.forward(x)
        >>> # output: GELU applied separately to real and imaginary parts

    """
    ellipsis_axes = EllipsisAxes()

    def __init__(
        self,
        name: str = "cgelu",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        """Initialize the Complex GELU activation function.

        Args:
            name: Name of the Node. Defaults to "cgelu".
            backend: Backend used for computation.
                Defaults to DEFAULT_DL_BACKEND.
        """

        self.reim = ReIm(backend=backend)
        self.cast = Cast(backend=backend)
        self.cast2 = CopiedNode(self.cast)
        self.gelu = GELU(backend=backend)
        self.gelu2 = CopiedNode(self.gelu)
        super().__init__(name, backend)

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of Complex GELU activation.

        Args:
            x: Complex-valued input tensor of any shape.

        Returns:
            TensorType: Complex-valued tensor with GELU applied independently
                to the real and imaginary parts, same shape as input.
        """

        re, im = self.reim(x)
        return self.cast(self.gelu(re), "complex64") + 1j * self.cast2(
            self.gelu2(im), "complex64"
        )
