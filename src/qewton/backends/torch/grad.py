import torch
from qewton.backends.grad import GradBackend


class TorchGradBackend(GradBackend[torch.Tensor]):
    """Torch implementations of differential operators."""

    @staticmethod
    def gradient_tracking(inp):
        inp.requires_grad = True
        return inp

    @staticmethod
    def gradient(u, x):
        return torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    @staticmethod
    def laplacian(u, x):
        laplacian = torch.zeros((*u.shape[:-1], 1), device=u.device)

        # Compute first derivative
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        # If linear w.r.t. x, gradient has no grad_fn, return zeros
        if grad.grad_fn is None:
            return laplacian

        # Sum second derivatives
        for i in range(x.shape[-1]):
            d2u = torch.autograd.grad(grad.narrow(-1, i, 1).sum(), x, create_graph=True)[
                0
            ]
            laplacian += d2u.narrow(-1, i, 1)

        return laplacian

    @staticmethod
    def normal_derivative(u, normals, x):
        # Compute gradient
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        # Compute normal derivative as gradient · normals
        normal_deriv = (grad * normals).sum(dim=-1, keepdim=True)

        return normal_deriv

    @staticmethod
    def divergence(u, x):
        divergence = torch.zeros((*x.shape[:-1], 1), device=u.device)

        # For each component of the output, compute du_i/dx_i and sum
        for i in range(u.shape[-1]):
            du_i = torch.autograd.grad(u.narrow(-1, i, 1).sum(), x, create_graph=True)[0]
            divergence += du_i.narrow(-1, i, 1)

        return divergence

    @staticmethod
    def jacobian(u, x):
        jac_rows = []

        # For each output dimension, compute derivatives w.r.t. all input dimensions
        for i in range(u.shape[-1]):
            du_i = torch.autograd.grad(u[..., i].sum(), x, create_graph=True)[0]
            jac_rows.append(du_i)

        # Stack to form jacobian matrix (batch, output_dim, input_dim)
        return torch.stack(jac_rows, dim=-2)

    @staticmethod
    def partial(u, x):
        # Compute partial derivative
        if u.grad_fn is None:
            return torch.zeros_like(x)

        return torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    @staticmethod
    def hessian(u, x):
        # Compute first derivative (gradient)
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        hessian_rows = []

        # For each component of the gradient, compute its derivative w.r.t. x
        for i in range(grad.shape[-1]):
            d2u = torch.autograd.grad(grad[..., i].sum(), x, create_graph=True)[0]
            hessian_rows.append(d2u)

        # Stack to form Hessian matrix (batch, input_dim, input_dim)
        return torch.stack(hessian_rows, dim=-2)
