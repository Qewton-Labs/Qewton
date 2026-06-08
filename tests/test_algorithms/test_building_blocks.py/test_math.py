import unittest
from unittest.mock import MagicMock, patch

from qewton.algorithms.building_blocks.math import (
    Add,
    Subtract,
    Mod,
    Square,
    Sqrt,
    Exp,
    Log,
    Log2,
    Log10,
    Sin,
    Cos,
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
from qewton.backends import Backend
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import EllipsisAxes


class MockBackend(Backend):
    library = MagicMock()

    @classmethod
    def standard_datatype(cls):
        return "float32"


class TestMathNodes(unittest.TestCase):
    def setUp(self):
        MockBackend.library.reset_mock()

    def test_arithmetic_forward_calls(self):
        # Test Add
        node = Add(backend=MockBackend)
        node.forward(1.0, 2.0)
        MockBackend.library.add.assert_called_with(1.0, 2.0)

        # Test Subtract (uses implementation routing)
        node = Subtract(backend=MockBackend)
        with patch.object(node, "implementation") as mock_impl:
            node.forward(5.0, 3.0)
            mock_impl.assert_called_with(5.0, 3.0)

    def test_powers_and_roots(self):
        node = Square(backend=MockBackend)
        node.forward(4.0)
        MockBackend.library.square.assert_called_with(4.0)

        node = Sqrt(backend=MockBackend)
        node.forward(16.0)
        MockBackend.library.sqrt.assert_called_with(16.0)

    def test_exponential_and_logs(self):
        node = Exp(backend=MockBackend)
        node.forward(1.0)
        MockBackend.library.exp.assert_called_with(1.0)

        node = Log(backend=MockBackend)
        node.forward(1.0)
        MockBackend.library.log.assert_called_with(1.0)

    def test_trigonometric_functions(self):
        node = Sin(backend=MockBackend)
        node.forward(0.0)
        MockBackend.library.sin.assert_called_with(0.0)

        node = Cos(backend=MockBackend)
        node.forward(0.0)
        MockBackend.library.cos.assert_called_with(0.0)

    def test_matrix_operations(self):
        # MatMul
        node = MatMul(backend=MockBackend)
        self.assertEqual(len(node.input_ports), 2)
        self.assertEqual(len(node.output_ports), 1)
        node.forward("matrix_a", "matrix_b")
        MockBackend.library.matmul.assert_called_with("matrix_a", "matrix_b")

        # SVD
        node = SVD(backend=MockBackend)
        self.assertEqual(len(node.output_ports), 3)  # U, S, V

    def test_reduction_node_logic(self):
        # Test Mean with axis reduction
        node = Mean(axis=0, keepdims=False, backend=MockBackend)

        # Mock DataConfiguration behavior for update_data_configs
        mock_in_config = MagicMock(spec=DC)
        mock_out_config = MagicMock(spec=DC)

        # Setup for _build_reduced_out_config
        mock_in_config.get_axes_and_dim.return_value = ("Feature", MagicMock())
        mock_in_config.update_config.return_value = True

        dynamic_configs = {
            node.input_ports[0]: mock_in_config,
            node.output_ports[0]: mock_out_config,
        }

        # Test update_data_configs
        with patch(
            "qewton.algorithms.building_blocks.math.deepcopy",
            return_value=mock_in_config,
        ):
            node.update_data_configs(
                node.input_ports[0], {"some": "update"}, dynamic_configs
            )
            mock_out_config.update_config.assert_called()

    def test_reduction_node_keepdims(self):
        node = Sum(axis=(0, 1), keepdims=True, backend=MockBackend)
        mock_in_config = MagicMock(spec=DC)
        mock_in_config.axes = [
            EllipsisAxes()
        ]  # Force None return in _build_keepdims_config
        mock_in_config.update_config.return_value = True

        dynamic_configs = {
            node.input_ports[0]: mock_in_config,
            node.output_ports[0]: MagicMock(),
        }

        with patch(
            "qewton.algorithms.building_blocks.math.deepcopy",
            return_value=mock_in_config,
        ):
            # Should return None/Empty set if ellipsis makes index counting impossible
            result = node.update_data_configs(node.input_ports[0], {}, dynamic_configs)
            self.assertEqual(len(result), 0)

    def test_stats_forward_routing(self):
        node = Std(axis=1, keepdims=True, backend=MockBackend)
        with patch.object(node, "implementation") as mock_impl:
            node.forward("tensor")
            mock_impl.assert_called_with("tensor")

    def test_flatten_and_transpose_init(self):
        # Flatten
        f_node = Flatten(start_dim=1, end_dim=2, backend=MockBackend)
        self.assertEqual(f_node.start_dim, 1)
        self.assertEqual(f_node.end_dim, 2)

        # Transpose
        t_node = Transpose(perm=[0, 2, 1], backend=MockBackend)
        self.assertEqual(t_node.perm, [0, 2, 1])

    def test_maximum_minimum(self):
        max_node = Maximum(backend=MockBackend)
        max_node.forward(10, 20)
        MockBackend.library.maximum.assert_called_with(10, 20)

        min_node = Minimum(backend=MockBackend)
        min_node.forward(10, 20)
        MockBackend.library.minimum.assert_called_with(10, 20)

    def test_log_variants_implementations(self):
        # Log2
        node = Log2(backend=MockBackend)
        node.torch_implementation("x")
        MockBackend.library.log2.assert_called_with("x")

        # Log10
        node = Log10(backend=MockBackend)
        node.torch_implementation("x")
        MockBackend.library.log10.assert_called_with("x")

    def test_trig_arcsin_arccos_arctan(self):
        node = ArcSin(backend=MockBackend)
        node.torch_implementation("x")
        MockBackend.library.arcsin.assert_called_with("x")

        node = ArcCos(backend=MockBackend)
        node.torch_implementation("x")
        MockBackend.library.arccos.assert_called_with("x")

        node = ArcTan(backend=MockBackend)
        node.torch_implementation("x")
        MockBackend.library.arctan.assert_called_with("x")

    def test_floor_ceil_abs(self):
        node = Floor(backend=MockBackend)
        node.forward(1.5)
        MockBackend.library.floor.assert_called_with(1.5)

        node = Ceil(backend=MockBackend)
        node.forward(1.1)
        MockBackend.library.ceil.assert_called_with(1.1)

        node = Abs(backend=MockBackend)
        node.forward(-5)
        MockBackend.library.abs.assert_called_with(-5)

    def test_mod_operation(self):
        node = Mod(backend=MockBackend)
        node.torch_implementation(10, 3)
        MockBackend.library.remainder.assert_called_with(10, 3)
        node.tensorflow_implementation(10, 3)
        MockBackend.library.mod.assert_called_with(10, 3)
