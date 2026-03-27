from pioneer.algorithms.implementation import DEFAULT_DL_IMPLEMENTATION, Implementation

from ..base import LayerNode
from ..implementation import (
    DEFAULT_DL_IMPLEMENTATION,
    TorchImplementation,
    TensorflowImplementation,
)
from ...nodes.base import NodeState
from ...nodes.base import InputPort
from ...config.configuration_base import DataConfiguration
from ...optim.parameters.trainable_parameters import TrainableParameters

# region: Base operation classes


# The following classes represent basic mathematical operations.
# They are designed to work with different operations and one only needs to pass
# in the name of the respective function in the backend to create a new operation.
# This reduces the number of classes we need to create. However, if there are
# specific operations that require special handling, we can of course
# create separate classes for them.
# TODO: Can the data configurations be handled by such a general class?
class TorchMathOperation(TorchImplementation):
    """Stateless math operation for PyTorch."""

    def __init__(self, function_name: str):
        import torch  # pylint: disable=import-error # type: ignore

        super().__init__(getattr(torch, function_name))


class TensorFlowMathOperation(TensorflowImplementation):
    """Stateless math operation for TensorFlow."""

    def __init__(self, function_name: str):
        import tensorflow as tf  # pylint: disable=import-error # type: ignore

        super().__init__(getattr(tf, function_name))


class SingleInputOperation(LayerNode):
    node_name = "Single Input Operation"

    operation_name = {  # overwritten in subclasses
        TorchImplementation: "",
        TensorflowImplementation: "",
    }
    existing_implementations = {
        TorchImplementation: TorchMathOperation,
        TensorflowImplementation: TensorFlowMathOperation,
    }

    def __init__(self, name=None, backend=DEFAULT_DL_IMPLEMENTATION):
        """A node representing a unary mathematical operation.

        Args:
            name (str, optional): Name of the Node.
            backend (Implementation, optional): The backend used for the
                mean computation. Defaults to DEFAULT_DL_IMPLEMENTATION.
        """
        if name is None:
            name = self.node_name
        super().__init__(name=name, backend=backend, state=NodeState.FIXED)
        self._input_ports[0].data_configuration.specify_dtype(backend)
        self._output_ports[0].data_configuration.specify_dtype(backend)
        # Only now do we instantiate the backend wrapper
        self.implementation_instance = self._build_implementation()

    def _build_implementation(self):
        return self.implementation(self.operation_name[self.backend])  # type: ignore

    @property
    def trainable_parameters(self):
        return TrainableParameters.create_empty(self.node_id)


class DoubleInputOperation(SingleInputOperation):
    node_name = "Double Input Operation"

    def __init__(self, name=None, backend=DEFAULT_DL_IMPLEMENTATION):
        """A node representing a mathematical operation that requires two
        input arguments. The port names are set to "input1" and "input2" to
        differentiate them.

        Args:
            name (str, optional): Name of the Node.
            backend (Implementation, optional): The backend used for the
                mean computation. Defaults to DEFAULT_DL_IMPLEMENTATION.
        """
        if name is None:
            name = getattr(self, "node_name", "Math Operation 2 Inputs")
        super().__init__(name=name, backend=backend)
        self._input_ports[0].name = "input1"
        self._input_ports.append(InputPort(DataConfiguration([]), self, "input2"))
        self._input_ports[1].data_configuration.specify_dtype(backend)

    def run(self):
        self._output_ports[0].set_value(
            self.implementation_instance(
                self._input_ports[0].value, self._input_ports[1].value
            )
        )


# endregion

# region: Arithmetic operations


class Add(DoubleInputOperation):
    node_name = "AddNode"
    operation_name = {
        TorchImplementation: "add",
        TensorflowImplementation: "add",
    }


class Subtract(DoubleInputOperation):
    node_name = "SubtractNode"
    operation_name = {
        TorchImplementation: "sub",
        TensorflowImplementation: "subtract",
    }


class Multiply(DoubleInputOperation):
    node_name = "MultiplyNode"
    operation_name = {
        TorchImplementation: "mul",
        TensorflowImplementation: "multiply",
    }


class Divide(DoubleInputOperation):
    node_name = "DivideNode"
    operation_name = {
        TorchImplementation: "true_divide",
        TensorflowImplementation: "truediv",
    }


class Mod(DoubleInputOperation):
    node_name = "ModuloNode"
    operation_name = {
        TorchImplementation: "remainder",
        TensorflowImplementation: "mod",
    }


# endregion


# region: Powers and roots


class Square(SingleInputOperation):
    node_name = "SquareNode"
    operation_name = {
        TorchImplementation: "square",
        TensorflowImplementation: "square",
    }


class Sqrt(SingleInputOperation):
    node_name = "SquareRootNode"
    operation_name = {
        TorchImplementation: "sqrt",
        TensorflowImplementation: "sqrt",
    }


class Power(DoubleInputOperation):
    node_name = "PowerNode"
    operation_name = {
        TorchImplementation: "pow",
        TensorflowImplementation: "pow",
    }


# endregion


# region: Exponential and logarithmic functions


class Exp(SingleInputOperation):
    node_name = "ExponentialNode"
    operation_name = {
        TorchImplementation: "exp",
        TensorflowImplementation: "exp",
    }


class Log(SingleInputOperation):
    node_name = "LogarithmNode"
    operation_name = {
        TorchImplementation: "log",
        TensorflowImplementation: "log",
    }


class Log2(SingleInputOperation):
    node_name = "Log2Node"
    operation_name = {
        TorchImplementation: "log2",
        TensorflowImplementation: "log",
    }


class Log10(SingleInputOperation):
    node_name = "Log10Node"
    operation_name = {
        TorchImplementation: "log10",
        TensorflowImplementation: "log10",
    }


# endregion


# region: Trigonometric functions


class Sin(SingleInputOperation):
    node_name = "SineNode"
    operation_name = {
        TorchImplementation: "sin",
        TensorflowImplementation: "sin",
    }


class Cos(SingleInputOperation):
    node_name = "CosineNode"
    operation_name = {
        TorchImplementation: "cos",
        TensorflowImplementation: "cos",
    }


class Tan(SingleInputOperation):
    node_name = "TangentNode"
    operation_name = {
        TorchImplementation: "tan",
        TensorflowImplementation: "tan",
    }


class ArcSin(SingleInputOperation):
    node_name = "ArcSineNode"
    operation_name = {
        TorchImplementation: "arcsin",
        TensorflowImplementation: "asin",
    }


class ArcCos(SingleInputOperation):
    node_name = "ArcCosineNode"
    operation_name = {
        TorchImplementation: "arccos",
        TensorflowImplementation: "acos",
    }


class ArcTan(SingleInputOperation):
    node_name = "ArcTangentNode"
    operation_name = {
        TorchImplementation: "arctan",
        TensorflowImplementation: "atan",
    }


# endregion


# region: Other useful math functions


class Abs(SingleInputOperation):
    node_name = "AbsoluteNode"
    operation_name = {
        TorchImplementation: "abs",
        TensorflowImplementation: "abs",
    }


class Floor(SingleInputOperation):
    node_name = "FloorNode"
    operation_name = {
        TorchImplementation: "floor",
        TensorflowImplementation: "floor",
    }


class Ceil(SingleInputOperation):
    node_name = "CeilNode"
    operation_name = {
        TorchImplementation: "ceil",
        TensorflowImplementation: "ceil",
    }


class Maximum(DoubleInputOperation):
    node_name = "MaximumNode"
    operation_name = {
        TorchImplementation: "maximum",
        TensorflowImplementation: "maximum",
    }


class Minimum(DoubleInputOperation):
    node_name = "MinimumNode"
    operation_name = {
        TorchImplementation: "minimum",
        TensorflowImplementation: "minimum",
    }


# endregion


# region: Matrix operations


class MatMul(DoubleInputOperation):
    node_name = "MatrixMultiplyNode"
    operation_name = {
        TorchImplementation: "matmul",
        TensorflowImplementation: "matmul",
    }


# endregion


# region: Reshaping operations


class TorchMean(TorchImplementation):
    def __init__(self, axis, keepdims=False):
        from torch import mean as torch_mean  # pylint: disable=no-name-in-module

        super().__init__(torch_mean)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, x):
        return self._torch_module(x, dim=self.axis, keepdim=self.keepdims)


class TensorFlowMean(TensorflowImplementation):
    def __init__(self, axis=None, keepdims=False):
        import tensorflow as tf  # pylint: disable=import-error # type: ignore

        super().__init__(tf.reduce_mean)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, x):
        return self._tf_layer(x, axis=self.axis, keepdims=self.keepdims)


class Mean(LayerNode):
    existing_implementations = {
        TorchImplementation: TorchMean,
        TensorflowImplementation: TensorFlowMean,
    }

    def __init__(
        self,
        name: str = "MeanNode",
        axis: int | list[int] | tuple[int] | None = 0,
        keepdims: bool = True,
        backend: Implementation = DEFAULT_DL_IMPLEMENTATION,
    ):
        """Computes the mean of the input over a given axis.

        Args:
            name (str, optional): Name of the Node. Defaults to "MeanNode".
            axis (int | list[int] | tuple[int] | None, optional):
                The axis over which the mean should be computed. In case
                of None the mean over all axes is computed. Defaults to 0.
            keepdims (bool, optional): If the mean computation should
                remove or keep the dimension/axis of the data. Defaults to True.
            backend (Implementation, optional): The backend used for the
                mean computation. Defaults to DEFAULT_DL_IMPLEMENTATION.
        """
        super().__init__(name, backend, NodeState.FIXED)
        self._input_ports[0].data_configuration.specify_dtype(backend)
        self._output_ports[0].data_configuration.specify_dtype(backend)

        self.implementation_instance = self.implementation(
            axis=axis, keepdims=keepdims  # type: ignore
        )

    @property
    def trainable_parameters(self):
        return TrainableParameters.create_empty(self.node_id)


# endregion
