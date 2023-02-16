import firedrake
from firedrake import Constant, inner, dot, tr, grad, div, dx, ds
import icepack


def power(**kwargs):
    # Get all the dynamical fields
    u, M, h = map(kwargs.get, ("velocity", "membrane_stress", "thickness"))

    # Get the parameters for the constitutive relation
    ε, τ = map(kwargs.get, ("viscous_yield_strain", "viscous_yield_stress"))
    n = Constant(kwargs["exponent"])
    A = ε / τ**n

    mesh = u.ufl_domain()
    d = mesh.geometric_dimension()

    M_2 = (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2
    M_n = M_2 if float(n) == 1 else M_2 ** ((n + 1) / 2)
    return 2 * h * A / (n + 1) * M_n * dx
