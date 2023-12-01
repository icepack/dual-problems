import ufl
import firedrake
from firedrake import Constant, inner, sym, grad, dx, ds, dS
import icepack
from .viscosity import power as viscous_power


__all__ = ["viscous_power", "boundary", "constraint"]


# Physical constants
ρ_I = Constant(icepack.constants.ice_density)
ρ_W = Constant(icepack.constants.water_density)
ρ = ρ_I * (1 - ρ_I / ρ_W)
g = Constant(icepack.constants.gravity)


def boundary(**kwargs):
    # Get all the dynamical fields and boundary conditions
    u, M, h = map(kwargs.get, ("velocity", "membrane_stress", "thickness"))
    outflow_ids = kwargs["outflow_ids"]

    # Get the unit outward normal vector to the terminus
    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)

    return 0.5 * ρ * g * h**2 * inner(u, ν) * ds(outflow_ids)


def constraint(**kwargs):
    u, M, h = map(kwargs.get, ("velocity", "membrane_stress", "thickness"))
    ε = sym(grad(u))
    return (-h * inner(M, ε) - inner(0.5 * ρ * g * grad(h**2), u)) * dx


def constraint_edges(**kwargs):
    u, h = map(kwargs.get, ("velocity", "thickness"))
    mesh = ufl.domain.extract_unique_domain(u)
    ν = firedrake.FacetNormal(mesh)
    u_ν = inner(u, ν)
    return 0.5 * ρ * g * (h("+")**2 * u_ν("+") + h("-")**2 * u_ν("-")) * dS

