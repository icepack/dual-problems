import subprocess
import numpy as np
from numpy import pi as π
import pygmsh
import firedrake
from firedrake import inner


def make_mesh(R, δx):
    geometry = pygmsh.built_in.Geometry()
    x1 = geometry.add_point([-R, 0, 0], lcar=δx)
    x2 = geometry.add_point([+R, 0, 0], lcar=δx)

    center1 = geometry.add_point([0, 0, 0], lcar=δx)
    center2 = geometry.add_point([0, -4 * R, 0], lcar=δx)

    arcs = [
        geometry.add_circle_arc(x1, center1, x2),
        geometry.add_circle_arc(x2, center2, x1),
    ]

    line_loop = geometry.add_line_loop(arcs)
    plane_surface = geometry.add_plane_surface(line_loop)

    physical_lines = [geometry.add_physical(arc) for arc in arcs]
    physical_surface = geometry.add_physical(plane_surface)

    with open("ice-shelf.geo", "w") as geo_file:
        geo_file.write(geometry.get_code())

    command = "gmsh -2 -format msh2 -v 0 -o ice-shelf.msh ice-shelf.geo"
    subprocess.run(command.split())

    return firedrake.Mesh("ice-shelf.msh")


def make_initial_data(mesh, R):
    inlet_angles = π * np.array([-3 / 4, -1 / 2, -1 / 3, -1 / 6])
    inlet_widths = π * np.array([1 / 8, 1 / 12, 1 / 24, 1 / 12])

    x = firedrake.SpatialCoordinate(mesh)

    u_in = 300
    h_in = 350
    hb = 100
    dh, du = 400, 250

    hs, us = [], []
    for θ, ϕ in zip(inlet_angles, inlet_widths):
        x0 = R * firedrake.as_vector((np.cos(θ), np.sin(θ)))
        v = -firedrake.as_vector((np.cos(θ), np.sin(θ)))
        L = inner(x - x0, v)
        W = x - x0 - L * v
        Rn = 2 * ϕ / π * R
        q = firedrake.max_value(1 - (W / Rn) ** 2, 0)
        hs.append(hb + q * ((h_in - hb) - dh * L / R))
        us.append(firedrake.exp(-4 * (W / R) ** 2) * (u_in + du * L / R) * v)

    h_expr = firedrake.Constant(hb)
    for h in hs:
        h_expr = firedrake.max_value(h, h_expr)

    u_expr = sum(us)
    return h_expr, u_expr
