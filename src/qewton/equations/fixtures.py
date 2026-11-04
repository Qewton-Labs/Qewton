"""Canonical problems used in tests, notebooks and documentation."""

from .model.constraints import Complementary, Eq
from .expr.api import div, dot, dt, grad, jump, lap, mean, shift, trace
from .expr.nodes import FunctionSymbol, Normal, Restrict
from .geometry.sets import (
    Boundary,
    Domain,
    Interface,
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


def stokes() -> Problem:
    """The bulk steady Stokes equations; boundary conditions are omitted.

    This fixture isolates the saddle-point coupling: momentum couples velocity and
    pressure, while incompressibility couples only velocity.
    """
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    velocity = FunctionSymbol(Variable("u", dim=2), x)
    pressure = FunctionSymbol(Variable("p", dim=1), x)
    force = FunctionSymbol(Variable("f", dim=2), x)
    return Problem(
        constraints=(
            Eq(-lap(velocity, x) + grad(pressure, x), force).on(omega),
            Eq(div(velocity, x), 0).on(omega),
        )
    )


def wave() -> Problem:
    """w_tt - c^2 lap w = f on Omega x I, Dirichlet on the boundary, w and w_t at 0."""
    x = Variable("x", dim=2)
    t = Variable("t", dim=1)
    omega = Domain("Omega", ambient=x)
    interval = Domain("I", ambient=t)
    side = Product((Boundary(omega), interval))
    initial = Product((omega, Slice(t, 0.0)))
    w = FunctionSymbol(Variable("w", dim=1), x * t)
    w0 = FunctionSymbol(Variable("w0"), x)
    w1 = FunctionSymbol(Variable("w1"), x)
    g = FunctionSymbol(Variable("g"), x * t)
    c = FunctionSymbol(Variable("c"), None)
    f = FunctionSymbol(Variable("f", dim=1), x * t)
    return Problem(
        constraints=(
            Eq(dt(dt(w, t), t) - c**2 * lap(w, x), f).on(Product((omega, interval))),
            Eq(Restrict(w, side), g).on(side),
            Eq(Restrict(w, initial), w0).on(initial),
            Eq(Restrict(dt(w, t), initial), w1).on(initial),
        ),
        time=t,
    )


def advection() -> Problem:
    """u_t + a . grad u = 0 on Omega x I, with inflow data on Gamma_in and u0 at t = 0."""
    x = Variable("x", dim=2)
    t = Variable("t", dim=1)
    omega = Domain("Omega", ambient=x)
    interval = Domain("I", ambient=t)
    inflow = Part("Gamma_in", Boundary(omega))
    inflow_in_time = Product((inflow, interval))
    initial = Product((omega, Slice(t, 0.0)))
    u = FunctionSymbol(Variable("u", dim=1), x * t)
    u0 = FunctionSymbol(Variable("u0"), x)
    g = FunctionSymbol(Variable("g"), x * t)
    a = FunctionSymbol(Variable("a", dim=2), None)
    return Problem(
        constraints=(
            Eq(dt(u, t) + dot(a, grad(u, x)), 0).on(Product((omega, interval))),
            Eq(Restrict(u, inflow_in_time), g).on(inflow_in_time),
            Eq(Restrict(u, initial), u0).on(initial),
        ),
        time=t,
    )


def obstacle() -> Problem:
    """-lap u - f >= 0, u - psi >= 0, and one of them is zero everywhere; u = g on the
    boundary."""
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    d_omega = Boundary(omega)
    u = FunctionSymbol(Variable("u", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    psi = FunctionSymbol(Variable("psi", dim=1), x)
    g = FunctionSymbol(Variable("g"), x)
    return Problem(
        constraints=(
            Complementary(-lap(u, x) - f, u - psi).on(omega),
            Eq(Restrict(u, d_omega), g).on(d_omega),
        ),
    )


def transmission() -> Problem:
    """-div(k grad u) = f with a different k on two subdomains, glued at an interface.

    The transmission conditions say that u and the normal flux k grad u . n do not jump.
    """
    x = Variable("x", dim=2)
    omega = Domain("Omega", ambient=x)
    left, right = Part("Omega_1", omega), Part("Omega_2", omega)
    gamma = Interface(left, right)
    u = FunctionSymbol(Variable("u", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    g = FunctionSymbol(Variable("g", dim=1), x)
    k1 = FunctionSymbol(Variable("k1", dim=1), x)
    k2 = FunctionSymbol(Variable("k2", dim=1), x)

    def flux(k, side):
        return dot(trace(k * grad(u, x), gamma, side), Normal(gamma))

    return Problem(
        constraints=(
            Eq(-div(k1 * grad(u, x), x), f).on(left),
            Eq(-div(k2 * grad(u, x), x), f).on(right),
            Eq(jump(u, gamma), 0).on(gamma),
            Eq(flux(k2, 1), flux(k1, 0)).on(gamma),
            Eq(Restrict(u, Boundary(omega)), g).on(Boundary(omega)),
        ),
        partitions=(Partition(omega, (left, right)),),
        interfaces=(gamma,),
    )
