import torch

from qewton.backends import TorchBackend
from qewton.config.variables import Variable
from qewton.graphs.control_nodes.wrapper_node import FunctionWrappingNode


class TestFunctionWrappingNode:
    def test_builds_one_input_port_per_variable_annotated_parameter(self):
        U, X = Variable("u", 1), Variable("x", 1)

        def residual(u: U, x: X):
            return u - x

        node = FunctionWrappingNode(residual, backend=TorchBackend)
        assert [p.name for p in node.input_ports] == ["u", "x"]

    def test_gets_a_single_fallback_output_port_when_unannotated(self):
        """No return annotation means _build_ports alone would give zero
        output ports (see PlotNode's own docstring for the same subtlety) -
        FunctionWrappingNode compensates since it can't trace a shape from
        an eagerly-executed function anyway."""

        def residual(u, x):
            return u - x

        node = FunctionWrappingNode(residual, backend=TorchBackend)
        assert len(node.output_ports) == 1

    def test_executes_the_function_eagerly_with_real_tensors(self):
        """Uses control flow and a raw torch call that a TrackingObject-based
        tracer (FromFunctionNode) could never trace: a Python `if` branching
        on a real tensor's value, and torch.where with no Node wrapping it
        at all - this is exactly the "any non-node logic" track_residual is
        meant to allow."""
        U, X = Variable("u", 1), Variable("x", 1)

        def residual(u: U, x: X):
            if x.mean().item() > 0:
                return torch.where(u > 0, u, -u)
            return u

        node = FunctionWrappingNode(residual, backend=TorchBackend)
        u = torch.tensor([[-1.0], [2.0]])
        x = torch.tensor([[1.0], [1.0]])
        result = node(u, x)
        assert torch.equal(result, torch.tensor([[1.0], [2.0]]))

    def test_forward_is_the_original_function_not_a_traced_reconstruction(self):
        calls = []

        def residual(u):
            calls.append(u)
            return u * 2

        node = FunctionWrappingNode(residual, backend=TorchBackend)
        u = torch.tensor([1.0, 2.0])
        result = node(u)
        assert calls == [u]
        assert torch.equal(result, u * 2)
