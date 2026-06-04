import torch


from qewton.config.backend import TorchBackend
from qewton.algorithms.building_blocks.derivatives import (
    Gradient,
    Laplacian,
    Divergence,
    NormalDerivative,
    Jacobian,
    Partial,
    Hessian,
)


def test_grad_simple():
    x = torch.linspace(0, 1, 100, requires_grad=True)
    u = x**2 + 5.0
    grad_node = Gradient(backend=TorchBackend)
    u_x = grad_node(u, x)
    assert torch.allclose(u_x, 2 * x)


def test_grad_vector_input():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    u = (x**2).sum(dim=-1)
    grad_node = Gradient(backend=TorchBackend)
    u_x = grad_node(u, x)
    assert torch.allclose(u_x, 2 * x)


def test_grad_batched_polynomial():
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0]], requires_grad=True)
    u = x[:, 0] ** 3 + 2 * x[:, 1]
    grad_node = Gradient(backend=TorchBackend)
    u_x = grad_node(u, x)
    expected = torch.stack([3 * x[:, 0] ** 2, torch.full_like(x[:, 1], 2.0)], dim=-1)
    assert torch.allclose(u_x, expected)


def test_laplacian_quadratic_1d():
    x = torch.linspace(-1.0, 1.0, 10, requires_grad=True).view(-1, 1)
    u = x**2
    lap_node = Laplacian(backend=TorchBackend)
    lap = lap_node(u, x)
    assert torch.allclose(lap, torch.full_like(lap, 2.0))


def test_laplacian_quadratic_2d():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    u = (x**2).sum(dim=-1, keepdim=True)
    lap_node = Laplacian(backend=TorchBackend)
    lap = lap_node(u, x)
    assert torch.allclose(lap, torch.full_like(lap, 4.0))


def test_laplacian_linear_returns_zero():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    u = x.sum(dim=-1, keepdim=True)
    lap_node = Laplacian(backend=TorchBackend)
    lap = lap_node(u, x)
    assert torch.allclose(lap, torch.zeros_like(lap))


def test_divergence_simple_vector_field():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = torch.stack([x[..., 0], 2 * x[..., 1]], dim=-1)
    div_node = Divergence(backend=TorchBackend)
    divergence = div_node(u, x)
    assert torch.allclose(divergence, torch.tensor([[3.0]]))


def test_divergence_quadratic_field():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = torch.stack([x[..., 0] ** 2, x[..., 1] ** 2], dim=-1)
    div_node = Divergence(backend=TorchBackend)
    divergence = div_node(u, x)
    assert torch.allclose(divergence, torch.tensor([[6.0]]))


def test_divergence_batched_field():
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0]], requires_grad=True)
    u = torch.stack([x[..., 0] ** 2, x[..., 1] ** 2], dim=-1)
    div_node = Divergence(backend=TorchBackend)
    divergence = div_node(u, x)
    expected = torch.tensor([[6.0], [-1.0]])
    assert torch.allclose(divergence, expected)


def test_normal_derivative_unit_normal():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = (x**2).sum(dim=-1, keepdim=True)
    normals = torch.tensor([[1.0, 0.0]])
    normal_node = NormalDerivative(backend=TorchBackend)
    normal_deriv = normal_node(u, normals, x)
    assert torch.allclose(normal_deriv, torch.tensor([[2.0]]))


def test_normal_derivative_arbitrary_normal():
    x = torch.tensor([[1.0, 1.0]], requires_grad=True)
    u = x[..., 0] ** 2 + 3 * x[..., 1] ** 2
    normals = torch.tensor([[1.0, 1.0]]) / torch.sqrt(torch.tensor(2.0))
    normal_node = NormalDerivative(backend=TorchBackend)
    normal_deriv = normal_node(u, normals, x)
    expected = (torch.tensor([2.0, 6.0]) * normals).sum().view(1, 1)
    assert torch.allclose(normal_deriv, expected)


def test_normal_derivative_batched_normals():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    u = (x**2).sum(dim=-1, keepdim=True)
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    normal_node = NormalDerivative(backend=TorchBackend)
    normal_deriv = normal_node(u, normals, x)
    assert torch.allclose(normal_deriv, torch.tensor([[2.0], [2.0]]))


def test_jacobian_simple_2d():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    u = torch.stack([x[..., 0] ** 2, 3 * x[..., 1]], dim=-1)
    jac_node = Jacobian(backend=TorchBackend)
    jacobian = jac_node(u, x)
    expected = torch.tensor([[[2.0, 0.0], [0.0, 3.0]], [[6.0, 0.0], [0.0, 3.0]]])
    assert torch.allclose(jacobian, expected)


def test_jacobian_three_output_dimensions():
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    u = torch.stack([x[..., 0] ** 2, x[..., 1] * x[..., 2], x[..., 2] ** 3], dim=-1)
    jac_node = Jacobian(backend=TorchBackend)
    jacobian = jac_node(u, x)
    expected = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 2.0], [0.0, 0.0, 27.0]]])
    assert torch.allclose(jacobian, expected)


def test_jacobian_batched_linear_and_quadratic():
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)
    u = torch.stack([x[..., 0] + x[..., 1], x[..., 0] * x[..., 1]], dim=-1)
    jac_node = Jacobian(backend=TorchBackend)
    jacobian = jac_node(u, x)
    expected = torch.tensor([[[1.0, 1.0], [2.0, 1.0]], [[1.0, 1.0], [3.0, 2.0]]])
    assert torch.allclose(jacobian, expected)


def test_partial_scalar_function():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = x[..., 0] ** 2 + 3 * x[..., 1]
    partial_node = Partial(backend=TorchBackend)
    partial_deriv = partial_node(u, x)
    assert torch.allclose(partial_deriv, torch.tensor([[2.0, 3.0]]))


def test_partial_vector_sum_function():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = torch.stack([x[..., 0] ** 2, x[..., 1] + 5.0], dim=-1)
    partial_node = Partial(backend=TorchBackend)
    partial_deriv = partial_node(u, x)
    assert torch.allclose(partial_deriv, torch.tensor([[2.0, 1.0]]))


def test_partial_non_differentiable_returns_zero():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = torch.ones((1, 1))
    partial_node = Partial(backend=TorchBackend)
    partial_deriv = partial_node(u, x)
    assert torch.allclose(partial_deriv, torch.zeros_like(x))


def test_hessian_quadratic():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    u = x[..., 0] ** 2 + 3 * x[..., 1] ** 2
    hess_node = Hessian(backend=TorchBackend)
    hessian = hess_node(u, x)
    expected = torch.tensor([[[2.0, 0.0], [0.0, 6.0]]])
    assert torch.allclose(hessian, expected)


def test_hessian_mixed_cross_terms():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    u = x[..., 0] * x[..., 1]
    hess_node = Hessian(backend=TorchBackend)
    hessian = hess_node(u, x)
    expected = torch.tensor([[[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
    assert torch.allclose(hessian, expected)


def test_hessian_batched_quadratic():
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=True)
    u = x[..., 0] ** 2 + 4 * x[..., 1] ** 2
    hess_node = Hessian(backend=TorchBackend)
    hessian = hess_node(u, x)
    expected = torch.tensor([[[2.0, 0.0], [0.0, 8.0]], [[2.0, 0.0], [0.0, 8.0]]])
    assert torch.allclose(hessian, expected)
