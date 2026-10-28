"""Canonical problems used in tests, notebooks and documentation."""

from .constraints import Eq
from .expr.api import div, dot, dt, grad, lap
from .expr.nodes import FunctionSymbol, Normal, Restrict
from .geometry.sets import Boundary, Domain, Part, Partition, Product, Slice
from .model import Problem
from ..config.variables import Variable


def poisson() -> Problem:
    """-div(nu grad u) = f on Omega, Dirichlet on Gamma_D, Neumann on Gamma_N."""
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    d_omega = Boundary(omega)
    gamma_d, gamma_n = Part("Gamma_D", d_omega), Part("Gamma_N", d_omega)
    u = FunctionSymbol(Variable("u"), x)
    f = FunctionSymbol(Variable("f"), None)
    g = FunctionSymbol(Variable("g"), x)
    h = FunctionSymbol(Variable("h"), x)
    nu = FunctionSymbol(Variable("nu"), None)
    flux = nu * grad(u, x)
    return Problem(
        constraints=(
            Eq(-div(flux, x), f).on(omega),
            Eq(Restrict(u, gamma_d), g).on(gamma_d),
            Eq(dot(Restrict(flux, gamma_n), Normal(d_omega)), h).on(gamma_n),
        ),
        partitions=(Partition(d_omega, (gamma_d, gamma_n)),),
    )


def heat() -> Problem:
    """w_t - lap w = f on Omega x I, with initial data w0 at t = 0."""
    x = Variable("x", dim=2)
    t = Variable("t", dim=1)
    omega = Domain("Omega", ambient=x)
    space_time = Product((omega, Domain("I", ambient=t)))
    initial = Product((omega, Slice(t, 0.0)))
    w = FunctionSymbol(Variable("w"), x * t)
    w0 = FunctionSymbol(Variable("w0"), x)
    f = FunctionSymbol(Variable("f"), None)
    return Problem(
        constraints=(
            Eq(dt(w, t) - lap(w, x), f).on(space_time),
            Eq(Restrict(w, initial), w0).on(initial),
        ),
        time=t,
    )
