# https://github.com/BorisTheBrave/mc-dc
"""Provides a function for performing 3D Dual Countouring"""
import math
from itertools import product, repeat, chain
from multiprocessing import Pool

import numpy as np

from Globals import *
from mcdc.common import adapt
from mcdc.qef import solve_qef_3d
from mcdc.settings import ADAPTIVE
from mcdc.utils_3d import V3, Quad, Mesh


def dual_contour_3d(f, f_normal, ijk_to_xyz, xmax, ymax, zmax):
    """Iterates over a cells of size one between the specified range, and evaluates f and f_normal to produce
        a boundary by Dual Contouring. Returns a Mesh object."""
    # For each cell, find the best vertex for fitting f

    xmin = ymin = zmin = 0

    with Pool(n_process) as pool:
        inputs = product([f], range(xmin, xmax), range(ymin, ymax), range(zmin, zmax))
        all_changes = pool.starmap(dual_contour_3d_find_changes, inputs)
        xyzs = product(range(xmin, xmax), range(ymin, ymax), range(zmin, zmax))
        xyzs_with_changes, changes = zip(*((xyz, changes) for xyz, changes in zip(xyzs, all_changes) if len(changes) > 1))

        f_normal.calc_grads(list(chain(*changes)))

        inputs = zip(xyzs_with_changes, changes, repeat(f_normal))
        vert_indices = {xyz: i for i, xyz in enumerate(xyzs_with_changes)}  # 1 based
        vert_array = ijk_to_xyz(np.array(pool.starmap(get_vert, inputs)))  # convert vert to lay in [-1,1]^3

        if len(vert_array) == 0:
            raise Exception("Level Set is empty")

        inputs = product([f], [vert_indices], range(xmin, xmax), range(ymin, ymax), range(zmin, zmax))
        faces = list(chain(*pool.starmap(get_faces_per_xyz, inputs)))

    return Mesh(vert_array, faces)


def get_changes_vals(f, x, y, z):
    # Evaluate f at each corner
    v = np.empty((2, 2, 2))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                v[dx, dy, dz] = f(x + dx, y + dy, z + dz)

    # For each edge, identify where there is a sign change.
    # There are 4 edges along each of the three axes
    ratio = []
    changes_vals = []
    for dx in (0, 1):
        for dy in (0, 1):
            if (v[dx, dy, 0] > 0) != (v[dx, dy, 1] > 0):
                changes_vals.append((v[dx, dy, 0], v[dx, dy, 1]))
                ratio.append(adapt(v[dx, dy, 0], v[dx, dy, 1]))

    for dx in (0, 1):
        for dz in (0, 1):
            if (v[dx, 0, dz] > 0) != (v[dx, 1, dz] > 0):
                changes_vals.append((v[dx, 0, dz], v[dx, 1, dz]))
                ratio.append(adapt(v[dx, 0, dz], v[dx, 1, dz]))

    for dy in (0, 1):
        for dz in (0, 1):
            if (v[0, dy, dz] > 0) != (v[1, dy, dz] > 0):
                changes_vals.append((v[0, dy, dz], v[1, dy, dz]))
                ratio.append(adapt(v[0, dy, dz], v[1, dy, dz]))

    return ratio


def dual_contour_3d_find_changes(f, x, y, z):
    # Evaluate f at each corner
    v = np.empty((2, 2, 2))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                v[dx, dy, dz] = f(x + dx, y + dy, z + dz)

    # For each edge, identify where there is a sign change.
    # There are 4 edges along each of the three axes
    changes = []
    for dx in (0, 1):
        for dy in (0, 1):
            if (v[dx, dy, 0] > 0) != (v[dx, dy, 1] > 0):
                changes.append((x + dx, y + dy, z + adapt(v[dx, dy, 0], v[dx, dy, 1])))

    for dx in (0, 1):
        for dz in (0, 1):
            if (v[dx, 0, dz] > 0) != (v[dx, 1, dz] > 0):
                changes.append((x + dx, y + adapt(v[dx, 0, dz], v[dx, 1, dz]), z + dz))

    for dy in (0, 1):
        for dz in (0, 1):
            if (v[0, dy, dz] > 0) != (v[1, dy, dz] > 0):
                changes.append((x + adapt(v[0, dy, dz], v[1, dy, dz]), y + dy, z + dz))

    return changes


def get_vert(xyz, changes, f_normal):
    # For each sign change location v[i], we find the normal n[i].
    # The error term we are trying to minimize is sum( dot(x-v[i], n[i]) ^ 2)
    # In other words, minimize || A * x - b || ^2 where A and b are a matrix and vector
    # derived from v and n
    if ADAPTIVE:
        normals = [f_normal(*v) for v in changes]
        vert = solve_qef_3d(*xyz, changes, normals)
    else:
        vert = np.array(xyz) + 0.5
    return vert


def get_faces_per_xyz(f, vert_indices, x, y, z, xmin=0, ymin=0, zmin=0):
    # For each cell edge, emit a face between the center of the adjacent cells if it is a sign changing edge

    faces_per_xyz = []
    if x > xmin and y > ymin:
        solid1 = f(x, y, z + 0) > 0
        solid2 = f(x, y, z + 1) > 0
        if solid1 != solid2:
            faces_per_xyz.append(Quad(
                vert_indices[(x - 1, y - 1, z)],
                vert_indices[(x - 0, y - 1, z)],
                vert_indices[(x - 0, y - 0, z)],
                vert_indices[(x - 1, y - 0, z)],
            ).swap(solid2))
    if x > xmin and z > zmin:
        solid1 = f(x, y + 0, z) > 0
        solid2 = f(x, y + 1, z) > 0
        if solid1 != solid2:
            faces_per_xyz.append(Quad(
                vert_indices[(x - 1, y, z - 1)],
                vert_indices[(x - 0, y, z - 1)],
                vert_indices[(x - 0, y, z - 0)],
                vert_indices[(x - 1, y, z - 0)],
            ).swap(solid1))
    if y > ymin and z > zmin:
        solid1 = f(x + 0, y, z) > 0
        solid2 = f(x + 1, y, z) > 0
        if solid1 != solid2:
            faces_per_xyz.append(Quad(
                vert_indices[(x, y - 1, z - 1)],
                vert_indices[(x, y - 0, z - 1)],
                vert_indices[(x, y - 0, z - 0)],
                vert_indices[(x, y - 1, z - 0)],
            ).swap(solid2))
    return faces_per_xyz


def circle_function(x, y, z):
    return 2.5 - math.sqrt(x*x + y*y + z*z)


def circle_normal(x, y, z):
    l = math.sqrt(x*x + y*y + z*z)
    return V3(-x / l, -y / l, -z / l)


def intersect_function(x, y, z):
    y -= 0.3
    x -= 0.5
    x = abs(x)
    return min(x - y, x + y)


def normal_from_function(f, d=0.01):
    """Given a sufficiently smooth 3d function, f, returns a function approximating of the gradient of f.
    d controls the scale, smaller values are a more accurate approximation."""
    def norm(x, y, z):
        return V3(
            (f(x + d, y, z) - f(x - d, y, z)) / 2 / d,
            (f(x, y + d, z) - f(x, y - d, z)) / 2 / d,
            (f(x, y, z + d) - f(x, y, z - d)) / 2 / d,
        ).normalize()
    return norm


__all__ = ["dual_contour_3d"]
