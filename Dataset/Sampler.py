import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from trimesh import Trimesh
from trimesh.sample import sample_surface

from Globals import args, INSIDE_LABEL, OUTSIDE_LABEL


def _get_points_on_plane(plane, pca_projected_vertices, refinment_level):
    radius = 0.5 ** args.sampling_radius_exp[refinment_level]
    n_samples = args.n_samples[refinment_level]

    xys_around_edges, xys_on_edge = _sample_around_edges(plane, pca_projected_vertices, n_samples, radius)

    xys_around_vert = _sample_around_verts(n_samples, pca_projected_vertices, radius)

    noise = 2 * np.random.random_sample((args.n_white_noise, 2)) - 1

    return np.concatenate((xys_around_vert, xys_around_edges, noise)), xys_on_edge


def _sample_around_verts(n_samples, pca_projected_vertices, radius):
    thetas = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    points_on_unit_spere = radius * np.stack((np.cos(thetas), np.sin(thetas))).T
    xys_around_vert = np.empty((0, 2))
    xyz_on_vert = pca_projected_vertices
    for vert in xyz_on_vert:
        xys_around_vert = np.concatenate((xys_around_vert, points_on_unit_spere + vert))
    return xys_around_vert


def _sample_around_edges(plane, pca_projected_vertices, n_samples, radius):
    if n_samples == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    edges_2d = pca_projected_vertices[plane.edges]
    edges_directions = edges_2d[:, 0, :] - edges_2d[:, 1, :]
    edge_normals = edges_directions @ np.array([[0, 1], [-1, 0]])
    edge_normals /= np.linalg.norm(edge_normals, axis=1)[:, None]
    dist = np.linspace(0, 1, n_samples, endpoint=False)
    xys_around_edges = np.empty((0, 2))
    xys_on_edge = np.empty((0, 2))
    for edge, normal in zip(edges_2d, edge_normals):
        points_on_edge = np.array([d * edge[0] + (1 - d) * edge[1] for d in dist])

        xys_around_edges = np.concatenate(
            (xys_around_edges, points_on_edge + normal * radius, points_on_edge - normal * radius))
        xys_on_edge = np.concatenate((xys_on_edge, points_on_edge))
    return xys_around_edges, xys_on_edge


def _get_labeler(plane, pca_projected_vertices):
    shape_vertices = []
    shape_codes = []
    hole_vertices = []
    hole_codes = []
    for component in plane.connected_components:
        if not component.is_hole:
            # last vertex is ignored
            shape_vertices += list(pca_projected_vertices[component.vertices_indices]) + [[0, 0]]
            shape_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]
        else:
            # last vertex is ignored
            hole_vertices += list(pca_projected_vertices[component.vertices_indices]) + [[0, 0]]
            hole_codes += [Path.MOVETO] + [Path.LINETO] * (len(component) - 1) + [Path.CLOSEPOLY]

    # noinspection PyTypeChecker
    path = Path(shape_vertices, shape_codes)
    hole_path = Path(hole_vertices, hole_codes) if len(hole_vertices) > 0 else None

    def labler(xys):
        if path is None:
            return np.full(len(xys), OUTSIDE_LABEL)
        mask = path.contains_points(xys)
        # this does not handle non-empty holes!
        if hole_path is not None:
            pixels_in_hole = hole_path.contains_points(xys)
            mask &= np.logical_not(pixels_in_hole)
        labels = np.where(mask, np.full(mask.shape, INSIDE_LABEL), np.full(mask.shape, OUTSIDE_LABEL))
        return labels

    return labler


def sample_plane(plane, refinment_level):
    assert not plane.is_empty
    pca_projected_vertices, pca = plane.pca_projection  # should be on the plane

    labeler = _get_labeler(plane, pca_projected_vertices)

    xys_around_contour, xys_on_contour = _get_points_on_plane(plane, pca_projected_vertices, refinment_level)

    labels_around_contour = labeler(xys_around_contour)
    labels_on_contour = np.full(len(xys_on_contour), (INSIDE_LABEL + OUTSIDE_LABEL) / 2)

    return pca.inverse_transform(np.concatenate((xys_around_contour, xys_on_contour))), np.concatenate(
        (labels_around_contour, labels_on_contour))


def sample_hull(csl):
    hull = ConvexHull(csl.all_vertices * (1 + args.bounding_margin))
    boundary_xyzs = np.array(sample_surface(Trimesh(hull.points, hull.simplices), args.n_samples_boundary)[0])
    return boundary_xyzs
