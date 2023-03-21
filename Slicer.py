import argparse
import os

import numpy as np
import trimesh
from matplotlib.path import Path
from meshcut import cross_section
from shapely.geometry import LinearRing
from stl import mesh as mesh2
from tqdm import tqdm

from Dataset.CSL import CSL, ConnectedComponent, Plane
from Dataset.Helpers import plane_origin_from_params


def _get_verts_faces(filename):
    scene = trimesh.load_mesh(filename)

    verts = scene.vertices
    faces = scene.faces

    verts -= np.mean(verts, axis=0)
    scale = 1.1

    verts /= scale * np.max(np.absolute(verts))

    return verts, faces


def make_csl_from_mesh(filename, save_path, n_slices):
    verts, faces, = _get_verts_faces(filename)
    model_name = os.path.split(filename)[-1].split('.')[0]

    plane_normals, ds = _get_random_planes(n_slices)

    plane_normals = (plane_normals.T / np.linalg.norm(plane_normals.T, axis=0)).T
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]

    ccs_per_plane = [cross_section(verts, faces, plane_orig=o, plane_normal=n) for o, n in tqdm(list(zip(plane_origins, plane_normals)))]

    csl = _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane)

    _save_sliced_mesh(csl, faces, model_name, save_path, verts)
    return csl


def _save_sliced_mesh(csl, faces, model_name, save_path, verts):

    my_mesh = mesh2.Mesh(np.zeros(len(faces), dtype=mesh2.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = verts[f[j], :]

    my_mesh.save(os.path.join(save_path, f'{model_name}.stl'))
    csl.to_ply(os.path.join(save_path, f'{model_name}.ply'))
    csl.to_file(os.path.join(save_path, f'{model_name}.csl'))



def _get_random_planes(n_slices):
    plane_normals = np.random.randn(n_slices, 3)

    ds = -1 * (np.random.random_sample(n_slices) * 2 - 1)
    return plane_normals, ds


def _plane_from_mesh(ccs, plane_params, normal, origin, plane_id, csl):
    connected_components = []
    vertices = np.empty(shape=(0, 3))

    to_plane_cords = _get_to_plane_cords(ccs[0][0], normal, origin)

    for cc in ccs:
        # this does not handle non-empty holes
        if len(cc) > 2:
            is_hole, parent_cc_idx = _is_cc_hole(cc, ccs, to_plane_cords)

            oriented_cc = _orient_polyline(cc, is_hole, to_plane_cords)

            vert_start = len(vertices)
            if is_hole:
                connected_components.append(
                    ConnectedComponent(parent_cc_idx, 2, list(range(vert_start, vert_start + len(cc)))))
            else:
                connected_components.append(ConnectedComponent(-1, 1, list(range(vert_start, vert_start + len(cc)))))
            vertices = np.concatenate((vertices, oriented_cc))

    return Plane(plane_id, plane_params, vertices, connected_components, csl)


def _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane):
    def plane_gen(csl):
        planes = []
        i = 1

        for origin, normal, d, ccs in zip(plane_origins, plane_normals, ds, ccs_per_plane):
            plane_params = (*normal, d)

            if len(ccs) > 0:
                planes.append(_plane_from_mesh(ccs, plane_params, normal, origin, i, csl))
                i += 1
        return planes

    return CSL(model_name, plane_gen, n_labels=2)


def _get_to_plane_cords(point_on_plane, normal, origin):
    b0 = point_on_plane - origin
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(normal, b0)
    transformation_matrix = np.array([b0, b1])

    def to_plane_cords(xyzs):
        alinged = xyzs - origin
        return np.array([transformation_matrix @ v for v in alinged])

    return to_plane_cords


def _is_cc_hole(cc, ccs, transform):
    is_hole = False
    parent_cc_idx = None

    point_inside_cc = transform(cc[0:1])
    for i, other_cc in enumerate(ccs):
        if other_cc is cc:
            continue
        shape_vertices = list(transform(other_cc)) + [[0, 0]]
        shape_codes = [Path.MOVETO] + [Path.LINETO] * (len(other_cc) - 1) + [Path.CLOSEPOLY]
        path = Path(shape_vertices, shape_codes)
        if path.contains_points(point_inside_cc)[0]:
            # todo not necessarily 1 but enough for my purposes
            is_hole = True
            parent_cc_idx = i
            break
    return is_hole, parent_cc_idx


def _orient_polyline(verts, is_hole, to_plane_cords):
    if is_hole == LinearRing(to_plane_cords(verts)).is_ccw:
        oriented_verts = verts[::-1]
    else:
        oriented_verts = verts
    return oriented_verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slice a mesh')
    parser.add_argument('input', type=str, help='path to mesh')
    parser.add_argument('out_dir', type=str, help='out directory to save outputs')
    parser.add_argument('n_slices', type=int, help='n of slices')
    args = parser.parse_args()

    print(f'Slicing {args.input} with {args.n_slices} slices')
    csl = make_csl_from_mesh(args.input, args.out_dir, args.n_slices)

    print(f'Generated csl={csl.model_name} non-empty slices={len([p for p in csl.planes if not p.is_empty])}, n edges={len(csl)} '
          f'Artifacts at: {args.out_dir}')
