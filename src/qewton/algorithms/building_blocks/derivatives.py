from typing import Annotated


from qewton.config.axes import EllipsisAxes, FeatureAxes, AxesDim
from qewton.config.backend import TensorType
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.algorithms.backend_node import BackendNode

# TODO: Add Jax and Tensorflow implementations


class GradientTracking(BackendNode[TensorType]):
    ell_axes = EllipsisAxes()

    def forward(
        self,
        inp: Annotated[TensorType, DC(ell_axes)],
    ) -> Annotated[TensorType, DC(ell_axes)]:
        return self.implementation(inp)

    def torch_implementation(self, inp):
        inp.requires_grad = True
        return inp


class Gradient(BackendNode[TensorType]):
    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        return self.backend.library.autograd.grad(u.sum(), x, create_graph=True)[0]


class Laplacian(BackendNode[TensorType]):
    """Computes the laplacian (sum of second derivatives) of a scalar output
    with respect to the input variable."""

    ell_axes = EllipsisAxes()

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes())],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library
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


class NormalDerivative(BackendNode[TensorType]):
    """Computes the normal derivative (gradient · normals) of a scalar output."""

    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)
    u_dim = AxesDim(1)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(u_dim,)))],
        normals: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(u_dim,)))]:
        return self.implementation(u, normals, x)

    def torch_implementation(self, u, normals, x):
        torch = self.backend.library

        # Compute gradient
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        # Compute normal derivative as gradient · normals
        normal_deriv = (grad * normals).sum(dim=-1, keepdim=True)

        return normal_deriv


class Divergence(BackendNode[TensorType]):
    """Computes the divergence of a vector field (model output) with
    respect to spatial variables."""

    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library
        divergence = torch.zeros((*x.shape[:-1], 1), device=u.device)

        # For each component of the output, compute du_i/dx_i and sum
        for i in range(u.shape[-1]):
            du_i = torch.autograd.grad(u.narrow(-1, i, 1).sum(), x, create_graph=True)[0]
            divergence += du_i.narrow(-1, i, 1)

        return divergence


class Jacobian(BackendNode[TensorType]):
    """Computes the Jacobian matrix of a vector output with respect to input variables."""

    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim, x_dim)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library
        jac_rows = []

        # For each output dimension, compute derivatives w.r.t. all input dimensions
        for i in range(u.shape[-1]):
            du_i = torch.autograd.grad(u[..., i].sum(), x, create_graph=True)[0]
            jac_rows.append(du_i)

        # Stack to form jacobian matrix (batch, output_dim, input_dim)
        jacobian = torch.stack(jac_rows, dim=-2)

        return jacobian


class Partial(BackendNode[TensorType]):
    """Computes n-th order partial derivatives recursively with
    respect to multiple variables."""

    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library

        # Compute partial derivative
        if u.grad_fn is None:
            return torch.zeros_like(x)

        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        return du


class Hessian(BackendNode[TensorType]):
    """Computes the Hessian matrix (second partial derivatives) of a scalar
    output."""

    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim, x_dim)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library

        # Compute first derivative (gradient)
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

        hessian_rows = []

        # For each component of the gradient, compute its derivative w.r.t. x
        for i in range(grad.shape[-1]):
            d2u = torch.autograd.grad(grad[..., i].sum(), x, create_graph=True)[0]
            hessian_rows.append(d2u)

        # Stack to form Hessian matrix (batch, input_dim, input_dim)
        hessian = torch.stack(hessian_rows, dim=-2)

        return hessian


class MatrixDivergence(BackendNode[TensorType]):
    """Computes the row-wise divergence of a matrix-valued field.

    The output is a vector containing the divergence of each row of the
    matrix field with respect to the spatial input.
    """

    ell_axes = EllipsisAxes()
    u_dim = AxesDim(None)
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(u_dim, x_dim)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(u_dim,)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library

        matrix_div = torch.zeros((*u.shape[:-2], u.shape[-2]), device=u.device)
        for i in range(u.shape[-2]):
            row = u[..., i, :]
            row_div = torch.zeros((*row.shape[:-1], 1), device=u.device)

            for j in range(row.shape[-1]):
                grad_ij = torch.autograd.grad(row[..., j].sum(), x, create_graph=True)[0]
                row_div += grad_ij[..., j : j + 1]
            matrix_div[..., i : i + 1] = row_div

        return matrix_div


class Rotation(BackendNode[TensorType]):
    """Computes the curl/rotation of a 3D vector field."""

    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(3,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(3,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(3,)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library

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


class SymmetricGradient(BackendNode[TensorType]):
    """Computes the symmetric gradient 0.5 * (∇u + (∇u)^T)."""

    ell_axes = EllipsisAxes()
    u_dim = AxesDim(None)
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(u_dim,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(u_dim, x_dim)))]:
        return self.implementation(u, x)

    def torch_implementation(self, u, x):
        torch = self.backend.library

        jac_rows = [
            torch.autograd.grad(u[..., i].sum(), x, create_graph=True)[0]
            for i in range(u.shape[-1])
        ]
        jacobian = torch.stack(jac_rows, dim=-2)

        return 0.5 * (jacobian + jacobian.transpose(-2, -1))


# ...existing code...
