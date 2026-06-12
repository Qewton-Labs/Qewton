from typing import Annotated


from qewton.config.axes import EllipsisAxes, FeatureAxes, AxesDim
from qewton.backends import TensorType
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.algorithms.backend_node import BackendNode


class GradientTracking(BackendNode[TensorType]):
    ell_axes = EllipsisAxes()

    def forward(
        self,
        inp: Annotated[TensorType, DC(ell_axes)],
    ) -> Annotated[TensorType, DC(ell_axes)]:
        return self.backend.grad.gradient_tracking(inp)


class Gradient(BackendNode[TensorType]):
    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))]:
        return self.backend.grad.gradient(u, x)


class Laplacian(BackendNode[TensorType]):
    """Computes the laplacian (sum of second derivatives) of a scalar output
    with respect to the input variable."""

    ell_axes = EllipsisAxes()

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes())],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(1,)))]:
        return self.backend.grad.laplacian(u, x)


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
        return self.backend.grad.normal_derivative(u, normals, x)


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
        return self.backend.grad.divergence(u, x)


class Jacobian(BackendNode[TensorType]):
    """Computes the Jacobian matrix of a vector output with respect to input variables."""

    ell_axes = EllipsisAxes()
    x_dim = AxesDim(None)

    def forward(
        self,
        u: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
        x: Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim,)))],
    ) -> Annotated[TensorType, DC(ell_axes, FeatureAxes(shape=(x_dim, x_dim)))]:
        return self.backend.grad.jacobian(u, x)


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
        return self.backend.grad.partial(u, x)


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
        return self.backend.grad.hessian(u, x)
