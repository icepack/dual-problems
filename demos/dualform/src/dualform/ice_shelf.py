import firedrake
from firedrake import Constant, inner, dot, grad, div, dx, ds
import icepack
from .viscosity import power as viscous_power


__all__ = ["viscous_power", "boundary", "constraint"]


# Physical constants
ρ_I = Constant(icepack.constants.ice_density)
ρ_W = Constant(icepack.constants.water_density)
ρ = ρ_I * (1 - ρ_I / ρ_W)
g = Constant(icepack.constants.gravity)


def boundary(**kwargs):
    # Get all the dynamical fields
    u, M, h = map(kwargs.get, ("velocity", "membrane_stress", "thickness"))

    # Get the parameters for the constitutive relation
    parameter_names = (
        "viscous_yield_strain", "viscous_yield_stress", "flow_law_exponent"
    )
    ε, τ, n = map(kwargs.get, parameter_names)
    A = ε / τ**n

    # Get the boundary conditions
    bc_keys = ("velocity_in", "inflow_ids", "outflow_ids")
    u_in, inflow_ids, outflow_ids = map(kwargs.get, bc_keys)

    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)
    d = mesh.geometric_dimension()

    M_ν = dot(M, ν) - 0.5 * ρ * g * h * ν
    M_ν_2 = inner(M_ν, M_ν)
    M_ν_n = M_ν_2 if float(n) == 1 else M_ν_2 ** ((n + 1) / 2)

    inflow_boundary = h * inner(dot(M, ν), u_in) * ds(inflow_ids)
    outflow_boundary = inner(u, h * M_ν) * ds(outflow_ids)

    α = Constant(10.0)  # TODO: figure out what this really needs to be
    l = firedrake.CellSize(mesh)
    penalty = 2 * α * l / (n + 1) * h * A * M_ν_n * ds(outflow_ids)
    return penalty - inflow_boundary - outflow_boundary


def constraint(**kwargs):
    # Get all the dynamical fields
    u, M, h = map(kwargs.get, ("velocity", "membrane_stress", "thickness"))
    return inner(u, div(h * M) - 0.5 * ρ * g * grad(h**2)) * dx
