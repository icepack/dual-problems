import numpy as np
import firedrake
from firedrake import assemble, Constant, inner, grad, div, dx, ds


def default_params(Q, V, f, g, k, n):
    if isinstance(k, firedrake.Constant):
        k_degree = 0
    elif isinstance(k, firedrake.Function):
        k_degree = k.ufl_element().degree()

    u_degree = V.ufl_element().degree()
    p_degree = Q.ufl_element().degree()
    quadrature_degree = max(
        k_degree + u_degree * int(np.ceil(float(n)) + 1), p_degree + u_degree
    )

    return {
        "form_compiler_parameters": {"quadrature_degree": quadrature_degree},
        "solver_parameters": {
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }


def solve_nonlinear_laplace(Q, V, f, g, k, n, params={}):
    r"""Compute a solution of the nonlinear Laplace equation

    Parameters
    ----------
    Q : firedrake.FunctionSpace
        The function space for the primal variable, i.e. pressure
    V : firedrake.FunctionSpace
        The function space for the dual variable, i.e. velocity
    f : firedrake.Function
        The right-hand side or source term, i.e. water infiltration rate
    g : firedrake.Function
        The boundary values
    k : firedrake.Function
        The conductivity coefficient
    n : firedrake.Constant
        The exponent for the problem; `n = 1` gives a linear PDE
    params : dict
        Optional parameters to pass to `firedrake.solve`

    Returns
    -------
    p, u : firedrake.Function
        The pressure and velocity solutions
    """
    params = params or default_params(Q, V, f, g, k, n)

    mesh = Q.mesh()
    d = mesh.geometric_dimension()

    # Compute a natural length scale for the domain
    volume = assemble(Constant(1) * dx(mesh))
    surface = assemble(Constant(1) * ds(mesh))
    length = d * (volume / surface)

    # Compute a natural scale for the solution magnitude and its gradient
    k_rms = np.sqrt(assemble(k**2 * dx(mesh)) / volume)
    f_rms = np.sqrt(assemble(f**2 * dx(mesh)) / volume)
    g_rms = np.sqrt(assemble(g**2 * ds(mesh)) / surface)
    # TODO: Estimate for `g` too
    U = Constant(length**n * f_rms / k_rms)

    Z = Q * V
    z = firedrake.Function(Z)
    p, u = firedrake.split(z)
    ν = firedrake.FacetNormal(mesh)

    # Initial linear solve to spin up `z`
    energy = 1 / (2 * k**n) * U**(n - 1) * inner(u, u) * dx
    constraint = p * (div(u) - f) * dx
    boundary = g * inner(u, ν) * ds
    L0 = energy - constraint + boundary
    F0 = firedrake.derivative(L0, z)
    firedrake.solve(F0 == 0, z, **params)

    energy = 1 / ((n + 1) * k**n) * inner(u, u)**((n + 1) / 2) * dx
    constraint = p * (div(u) - f) * dx
    boundary = g * inner(u, ν) * ds
    L = energy - constraint + boundary
    F = firedrake.derivative(L, z)
    firedrake.solve(F == 0, z, **params)
    return z.split()


if __name__ == "__main__":
    R = Constant(2.0)

    n = Constant(3.0)
    k = Constant(1.5)
    f = Constant(0.75)
    g = Constant(0.0)

    # TODO: Make this go higher, fails at 8
    refinement_levels = list(range(2, 7))
    relative_errors = np.zeros(len(refinement_levels))

    for index, refinement_level in enumerate(refinement_levels):
        mesh = firedrake.UnitDiskMesh(refinement_level)
        x = firedrake.SpatialCoordinate(mesh)
        Vc = mesh.coordinates.function_space()
        X = firedrake.interpolate(R * x, Vc)
        mesh.coordinates.assign(X)

        cg1 = firedrake.FiniteElement("CG", "triangle", 1)
        Q = firedrake.FunctionSpace(mesh, cg1)
        b3 = firedrake.FiniteElement("B", "triangle", 3)
        V = firedrake.VectorFunctionSpace(mesh, cg1 + b3)

        r = inner(x, x)**0.5
        exact_solution = (f / (2 * k))**n * (R**(n + 1) - r**(n + 1)) / (n + 1)

        p, u = solve_nonlinear_laplace(Q, V, f, g, k, n)
        norm = assemble(abs(exact_solution)**(1 / n + 1) * dx)
        error = assemble(abs(p - exact_solution)**(1 / n + 1) * dx)
        relative_errors[index] = (error / norm) ** (n / (n + 1))
        print(".", end="", flush=True)

    slope, intercept = np.polyfit(refinement_levels, np.log2(relative_errors), 1)
    print(f"log(error) ~= {slope:g} * level {intercept:+g}")
