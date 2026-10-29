"""Canonical problems used in tests, notebooks and documentation."""

from .model.constraints import Eq
from .expr.api import div, dot, dt, grad, lap, mean, shift
from .expr.nodes import FunctionSymbol, Normal, Restrict
from .geometry.sets import (
    Boundary,
    Domain,
    Part,
    Partition,
    Periodic,
    Product,
    Slice,
)
from .model.problem import Problem
from ..config.variables import Variable


def poisson() -> Problem:
    """-div(nu grad u) = f on Omega, Dirichlet on Gamma_D, Neumann on Gamma_N."""
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    d_omega = Boundary(omega)
    gamma_d, gamma_n = Part("Gamma_D", d_omega), Part("Gamma_N", d_omega)
    u = FunctionSymbol(Variable("u", dim=1), x)
    f = FunctionSymbol(Variable("f"), None)
    g = FunctionSymbol(Variable("g"), x)
    h = FunctionSymbol(Variable("h"), x)
    nu = FunctionSymbol(Variable("nu", dim=1), None)
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


def neumann_poisson() -> Problem:
    """-lap p = f with a Neumann condition: p is only fixed up to a constant, so the
    mean of p is set to zero by a global constraint."""
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    d_omega = Boundary(omega)
    p = FunctionSymbol(Variable("p", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    h = FunctionSymbol(Variable("h", dim=1), x)
    return Problem(
        constraints=(
            Eq(-lap(p, x), f).on(omega),
            Eq(dot(Restrict(grad(p, x), d_omega), Normal(d_omega)), h).on(d_omega),
            Eq(mean(p, omega), 0).on(None),
        ),
    )


def delay() -> Problem:
    """u_t(t) = -u(t - 1), with history phi on the interval [-1, 0]."""
    t = Variable("t", dim=1)
    interval = Domain("I", ambient=t)
    history = Domain("H", ambient=t)
    u = FunctionSymbol(Variable("u", dim=1), t)
    phi = FunctionSymbol(Variable("phi", dim=1), t)
    return Problem(
        constraints=(
            Eq(dt(u, t), -shift(u, t, -1)).on(interval),
            Eq(Restrict(u, history), phi).on(history),
        ),
        time=t,
    )


def periodic_poisson(length: float = 2.0) -> Problem:
    """-lap u = f on a box, periodic in x_1 between Gamma_L and Gamma_R."""
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    d_omega = Boundary(omega)
    left, right = Part("Gamma_L", d_omega), Part("Gamma_R", d_omega)
    u = FunctionSymbol(Variable("u", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    offset = (length, 0.0)
    return Problem(
        constraints=(
            Eq(-lap(u, x), f).on(omega),
            Eq(Restrict(u, right), shift(Restrict(u, left), x, offset)).on(right),
        ),
        periodic=(Periodic(left, right, x, offset),),
    )
