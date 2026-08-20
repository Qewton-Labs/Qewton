import unittest
from unittest.mock import MagicMock, patch
import numpy as np

try:
    import torch
    from qewton.backends.torch.base import TorchBackend

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    from qewton.backends.tensorflow.base import TensorflowBackend

    HAS_TF = True
except ImportError:
    HAS_TF = False


from qewton.algorithms.building_blocks.math import (
    Add,
    Subtract,
    Multiply,
    Divide,
    Mod,
    Square,
    Sqrt,
    Power,
    Exp,
    Log,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    ArcSin,
    ArcCos,
    ArcTan,
    Abs,
    Floor,
    Ceil,
    Maximum,
    Minimum,
    MatMul,
    SVD,
    Mean,
    Sum,
    Std,
    Flatten,
    Transpose,
)
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import EllipsisAxes


class MockMathBackend:
    add = MagicMock()
    subtract = MagicMock()
    multiply = MagicMock()
    divide = MagicMock()
    mod = MagicMock()
    square = MagicMock()
    sqrt = MagicMock()
    power = MagicMock()
    exp = MagicMock()
    log = MagicMock()
    log2 = MagicMock()
    log10 = MagicMock()
    sin = MagicMock()
    cos = MagicMock()
    tan = MagicMock()
    arcsin = MagicMock()
    arccos = MagicMock()
    arctan = MagicMock()
    abs = MagicMock()
    floor = MagicMock()
    ceil = MagicMock()
    maximum = MagicMock()
    minimum = MagicMock()
    matmul = MagicMock()
    mean = MagicMock()
    sum = MagicMock()
    std = MagicMock()
    flatten = MagicMock()
    transpose = MagicMock()


class MockLinalgBackend:
    svd = MagicMock()


class MockBackend:
    math = MockMathBackend
    linalg = MockLinalgBackend
    default_dtype = "float32"


class TestMathNodes(unittest.TestCase):
    def setUp(self):
        # Reset all mocks in the classes
        for attr in dir(MockMathBackend):
            mock = getattr(MockMathBackend, attr)
            if isinstance(mock, MagicMock):
                mock.reset_mock()
        MockLinalgBackend.svd.reset_mock()

    def test_arithmetic_forward_calls(self):
        Add(backend=MockBackend).forward(1.0, 2.0)
        MockMathBackend.add.assert_called_with(1.0, 2.0)

        Subtract(backend=MockBackend).forward(5.0, 3.0)
        MockMathBackend.subtract.assert_called_with(5.0, 3.0)

        Multiply(backend=MockBackend).forward(2.0, 3.0)
        MockMathBackend.multiply.assert_called_with(2.0, 3.0)

        Divide(backend=MockBackend).forward(10.0, 2.0)
        MockMathBackend.divide.assert_called_with(10.0, 2.0)

        Mod(backend=MockBackend).forward(10.0, 3.0)
        MockMathBackend.mod.assert_called_with(10.0, 3.0)

    def test_powers_and_roots(self):
        Square(backend=MockBackend).forward(4.0)
        MockMathBackend.square.assert_called_with(4.0)

        Sqrt(backend=MockBackend).forward(16.0)
        MockMathBackend.sqrt.assert_called_with(16.0)

        a = torch.tensor(2.0)
        Power(backend=MockBackend).forward(a, 3.0)

    def test_exponential_and_logs(self):
        Exp(backend=MockBackend).forward(1.0)
        MockMathBackend.exp.assert_called_with(1.0)

        Log(backend=MockBackend).forward(1.0)
        MockMathBackend.log.assert_called_with(1.0)

        Log2(backend=MockBackend).forward(1.0)
        MockMathBackend.log2.assert_called_with(1.0)

        Log10(backend=MockBackend).forward(1.0)
        MockMathBackend.log10.assert_called_with(1.0)

    def test_trigonometric_functions(self):
        Sin(backend=MockBackend).forward(0.0)
        MockMathBackend.sin.assert_called_with(0.0)

        Cos(backend=MockBackend).forward(0.0)
        MockMathBackend.cos.assert_called_with(0.0)

        Tan(backend=MockBackend).forward(0.0)
        MockMathBackend.tan.assert_called_with(0.0)

        ArcSin(backend=MockBackend).forward(0.0)
        MockMathBackend.arcsin.assert_called_with(0.0)

        ArcCos(backend=MockBackend).forward(0.0)
        MockMathBackend.arccos.assert_called_with(0.0)

        ArcTan(backend=MockBackend).forward(0.0)
        MockMathBackend.arctan.assert_called_with(0.0)

    def test_matrix_operations(self):
        node = MatMul(backend=MockBackend)
        node.forward("matrix_a", "matrix_b")
        MockMathBackend.matmul.assert_called_with("matrix_a", "matrix_b")

        node = SVD(backend=MockBackend)
        node.forward("matrix")
        self.assertEqual(len(node.output_ports), 3)  # U, S, V
        MockLinalgBackend.svd.assert_called_with("matrix")

    def test_reduction_node_logic(self):
        node = Mean(axis=0, keepdims=False, backend=MockBackend)
        mock_in_config = MagicMock(spec=DC)
        mock_out_config = MagicMock(spec=DC)
        mock_in_config.get_axes_and_dim.return_value = ("Feature", MagicMock())
        mock_in_config.update_config.return_value = True
        dynamic_configs = {
            node.input_ports[0]: mock_in_config,
            node.output_ports[0]: mock_out_config,
        }
        with patch(
            "qewton.algorithms.building_blocks.math.deepcopy", return_value=mock_in_config
        ):
            node.update_data_configs(
                node.input_ports[0], {"some": "update"}, dynamic_configs
            )
            mock_out_config.update_config.assert_called()

    def test_reduction_node_keepdims(self):
        node = Sum(axis=(0, 1), keepdims=True, backend=MockBackend)
        mock_in_config = MagicMock(spec=DC)
        mock_in_config.axes = [EllipsisAxes()]
        mock_in_config.update_config.return_value = True
        dynamic_configs = {
            node.input_ports[0]: mock_in_config,
            node.output_ports[0]: MagicMock(),
        }
        with patch(
            "qewton.algorithms.building_blocks.math.deepcopy", return_value=mock_in_config
        ):
            result = node.update_data_configs(node.input_ports[0], {}, dynamic_configs)
            self.assertEqual(len(result), 0)

    def test_stats_forward_calls(self):
        Mean(axis=0, keepdims=False, backend=MockBackend).forward("tensor")
        MockMathBackend.mean.assert_called_with("tensor", axis=0, keepdims=False)

        Sum(axis=(1, 2), keepdims=True, backend=MockBackend).forward("tensor")
        MockMathBackend.sum.assert_called_with("tensor", axis=(1, 2), keepdims=True)

        Std(axis=1, keepdims=True, backend=MockBackend).forward("tensor")
        MockMathBackend.std.assert_called_with("tensor", axis=1, keepdims=True)

    def test_reshaping_forward_calls(self):
        Flatten(start_dim=1, end_dim=2, backend=MockBackend).forward("tensor")
        MockMathBackend.flatten.assert_called_with("tensor", start_dim=1, end_dim=2)

        Transpose(perm=[0, 2, 1], backend=MockBackend).forward("tensor")
        MockMathBackend.transpose.assert_called_with("tensor", axes=[0, 2, 1])

    def test_other_math_functions(self):
        Abs(backend=MockBackend).forward(-5)
        MockMathBackend.abs.assert_called_with(-5)

        Floor(backend=MockBackend).forward(1.5)
        MockMathBackend.floor.assert_called_with(1.5)

        Ceil(backend=MockBackend).forward(1.1)
        MockMathBackend.ceil.assert_called_with(1.1)

        Maximum(backend=MockBackend).forward(10, 20)
        MockMathBackend.maximum.assert_called_with(10, 20)

        Minimum(backend=MockBackend).forward(10, 20)
        MockMathBackend.minimum.assert_called_with(10, 20)


@unittest.skipIf(
    not (HAS_TORCH and HAS_TF),
    "Both Torch and TensorFlow are required for cross-backend tests",
)
class TestMathCrossBackend(unittest.TestCase):
    def to_numpy(self, val):
        if isinstance(val, (list, tuple)):
            return [self.to_numpy(v) for v in val]
        if hasattr(val, "detach"):  # Torch
            return val.detach().cpu().numpy()
        if hasattr(val, "numpy"):  # TF
            return val.numpy()
        return val

    def assert_backends_match(self, node_class, inputs, **node_kwargs):
        # inputs are numpy arrays
        x_torch = [TorchBackend.from_numpy(i) for i in inputs]
        x_tf = [TensorflowBackend.from_numpy(i) for i in inputs]

        node_torch = node_class(backend=TorchBackend, **node_kwargs)
        node_tf = node_class(backend=TensorflowBackend, **node_kwargs)

        out_torch = node_torch.forward(*x_torch)
        out_tf = node_tf.forward(*x_tf)

        res_torch = self.to_numpy(out_torch)
        res_tf = self.to_numpy(out_tf)

        if isinstance(res_torch, list):
            for rt, rtf in zip(res_torch, res_tf):
                np.testing.assert_allclose(rt, rtf, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(res_torch, res_tf, rtol=1e-5, atol=1e-5)

    def test_arithmetic(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        self.assert_backends_match(Add, [a, b])
        self.assert_backends_match(Subtract, [a, b])
        self.assert_backends_match(Multiply, [a, b])
        self.assert_backends_match(Divide, [a, b])

    def test_unary_and_transcendental(self):
        a = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
        self.assert_backends_match(Abs, [a])
        self.assert_backends_match(Exp, [a])
        self.assert_backends_match(Log, [a])
        self.assert_backends_match(Sin, [a])
        self.assert_backends_match(Cos, [a])

    def test_reductions(self):
        a = np.random.rand(4, 4).astype(np.float32)
        self.assert_backends_match(Mean, [a], axis=0)
        self.assert_backends_match(Sum, [a], axis=1)
        self.assert_backends_match(Std, [a], axis=None)

    def test_reshaping(self):
        a = np.random.rand(2, 3, 4).astype(np.float32)
        self.assert_backends_match(Flatten, [a], start_dim=1)
        self.assert_backends_match(Transpose, [a], perm=[2, 0, 1])
