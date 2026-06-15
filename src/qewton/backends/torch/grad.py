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

    @staticmethod
    def matrix_divergence(u, x):
        matrix_div = torch.zeros((*u.shape[:-2], u.shape[-2]), device=u.device)
        for i in range(u.shape[-2]):
            row = u[..., i, :]
            row_div = torch.zeros((*row.shape[:-1], 1), device=u.device)

            for j in range(row.shape[-1]):
                grad_ij = torch.autograd.grad(row[..., j].sum(), x, create_graph=True)[0]
                row_div += grad_ij[..., j : j + 1]
            matrix_div[..., i : i + 1] = row_div

        return matrix_div

    @staticmethod
    def rotation(u, x):
        if u.shape[-1] != 3 or x.shape[-1] != 3:
            raise ValueError("Rotation requires 3-dimensional field and input.")

        jac_rows = [
            torch.autograd.grad(u[..., i].sum(), x, create_graph=True)[0]
            for i in range(3)
        ]
        jacobian = torch.stack(jac_rows, dim=-2)

        rotation = torch.stack(
            [
                jacobian[..., 2, 1] - jacobian[..., 1, 2],
                jacobian[..., 0, 2] - jacobian[..., 2, 0],
                jacobian[..., 1, 0] - jacobian[..., 0, 1],
            ],
            dim=-1,
        )

        return rotation

    @staticmethod
    def symmetric_gradient(u, x):
        jac_rows = [
            torch.autograd.grad(u[..., i].sum(), x, create_graph=True)[0]
            for i in range(u.shape[-1])
        ]
        jacobian = torch.stack(jac_rows, dim=-2)

        return 0.5 * (jacobian + jacobian.transpose(-2, -1))
