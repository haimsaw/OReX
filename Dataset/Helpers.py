import numpy as np


def plane_origin_from_params(plane_params):
    normal = np.array(plane_params[0:3])
    d = plane_params[3]
    origin = normal * -1 * d
    return origin


def plane_d_from_origin(origin, normal):
    return -1 * sum([a * b for a, b in zip(origin, normal)])
