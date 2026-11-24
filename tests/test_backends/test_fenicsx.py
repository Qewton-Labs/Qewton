"""The FEniCSx backend: forms, solver reuse and submeshes.

Skipped unless dolfinx is installed (e.g. in the fenicsx conda environment).
"""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

# pylint: disable=wrong-import-position
from dolfinx import default_scalar_type, fem, mesh as dmesh
from mpi4py import MPI

from qewton.backends.fem.fenicsx.base import FenicsXBackend
from qewton.backends.fem.fenicsx.solve import FenicsXSolveBackend
from qewton.backends.fem.solve import LinearSolverOptions, NonlinearSolverOptions

B = FenicsXBackend


def exact(scale):
    """A quadratic solution of -lap u = -6 * scale, represented exactly by P2."""
    return lambda x: scale * (1 + x[0] ** 2 + 2 * x[1] ** 2)


def total(integrand, domain):
    """The integral of `integrand` over the whole mesh."""
    return B.form.assemble(B.form.integrate(integrand, B.mesh.volume_measure(domain)))


@pytest.fixture
def poisson():
    """A Poisson problem whose source and boundary values can be changed."""
    domain = dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = B.function_space.create_function_space(domain, "Lagrange", 2)
    u = B.function_space.create_trial_function(V)
    v = B.function_space.create_test_function(V)
    f = B.function_space.create_constant(default_scalar_type(-6.0), domain)
    u_bc = B.function_space.create_function(V)
    u_bc.interpolate(exact(1.0))
    bc = B.form.build_dirichlet(u_bc, lambda x: np.full(x.shape[1], True), V)

    dx = B.mesh.volume_measure(domain)
    a = B.form.integrate(B.form.inner(B.form.gradient(u), B.form.gradient(v)), dx)
    L = B.form.integrate(B.form.multiply(f, v), dx)
    uh = B.function_space.create_function(V)
    handle = B.solve.create_linear_solver(a, L, [bc], uh, LinearSolverOptions())
    return handle, uh, f, u_bc


def max_error(uh, scale):
    expected = fem.Function(uh.function_space)
    expected.interpolate(exact(scale))
    return np.max(np.abs(uh.x.array - expected.x.array))


def test_a_linear_solve_reports_its_convergence(poisson):
    handle, uh, _, _ = poisson
    result = B.solve.solve(handle)
    assert result.converged
    assert result.iterations >= 1
    assert max_error(uh, 1.0) < 1e-10


def test_a_new_right_hand_side_reuses_the_operator(poisson):
    handle, uh, f, u_bc = poisson
    assembled = []
    assemble_operator = handle._assemble_operator  # pylint: disable=protected-access

    def counting():
        assembled.append(True)
        assemble_operator()

    handle._assemble_operator = counting  # pylint: disable=protected-access

    B.solve.solve(handle, reassemble_operator=False)  # the first solve assembles
    B.function_space.set_constant_value(f, -12.0)
    u_bc.interpolate(exact(2.0))
    result = B.solve.solve(handle, reassemble_operator=False)

    assert len(assembled) == 1
    assert result.converged
    assert max_error(uh, 2.0) < 1e-10

    B.solve.solve(handle)
    assert len(assembled) == 2


def test_backend_options_take_precedence_for_linear_and_nonlinear_solvers():
    linear = LinearSolverOptions(backend_options={"pc_type": "cholesky"})
    assert FenicsXSolveBackend.translate_options(linear)["pc_type"] == "cholesky"
    nonlinear = NonlinearSolverOptions(
        linear=linear, backend_options={"snes_type": "newtontr"}
    )
    options = FenicsXSolveBackend.translate_options(nonlinear)
    assert options["pc_type"] == "cholesky"
    assert options["snes_type"] == "newtontr"


def test_derivative_is_the_partial_derivative_along_a_coordinate():
    domain = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = B.function_space.create_function_space(domain, "Lagrange", 2)
    w = B.function_space.create_function(V)
    w.interpolate(lambda x: x[1] ** 2)
    # d/dy y^2 = 2y integrates to 1 over the unit square, d/dx y^2 to 0.
    assert total(B.form.derivative(w, 1), domain) == pytest.approx(1.0)
    assert total(B.form.derivative(w, 0), domain) == pytest.approx(0.0, abs=1e-12)


def test_sin_cos_and_exp_build_forms():
    domain = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    zero = B.function_space.create_constant(default_scalar_type(0.0), domain)
    assert total(B.form.cos(zero), domain) == pytest.approx(1.0)
    assert total(B.form.exp(zero), domain) == pytest.approx(1.0)
    assert total(B.form.sin(zero), domain) == pytest.approx(0.0)


def test_get_subdomain_returns_the_submesh_and_its_entity_map():
    domain = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = domain.topology.dim
    tags = B.mesh.mark_mesh(domain, lambda x: x[0] <= 0.5 + 1e-12, 1, tdim)
    submesh, entity_map = B.mesh.get_subdomain(domain, tags, 1, tdim)
    n_sub = submesh.topology.index_map(tdim).size_local
    n_all = domain.topology.index_map(tdim).size_local
    assert 0 < n_sub < n_all
    assert entity_map is not None
