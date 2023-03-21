from itertools import chain
from os import path

from parse import parse
from sklearn.decomposition import PCA

from Dataset.Helpers import *
from Dataset.SlicesDataset import SlicesDataset

'''
CSLC # header
15 2  # number of planes, number of labels (should be at least 2 - inside and outside)

1 78 1 0.0 0.0 1.0 -0.86 # plane index (1-indexing, please state planes in order), number of vertices in the plane image (a hole is counted as another component), number of connected components, plane parameters A,B,C,D, such that Ax+By+Cz+D=0

0.1067530845 0.0856077309 0.8636363636 # The vertices in x,y,z coordinates, should be on the plane.
0.0930575569 0.0893916487 0.8636363636
0.0857102886 0.0920119983 0.8636363636
0.0728317043 0.0932992632 0.8636363636

[...] # rest of vertices

78 1 0 1 2 3 4 5 6 7 8 9 10 11 [...]  # image component: starts with the number of vertices, then label of the component (in case of a hole, h should be added and the index of the component contains the hole), then the indices of vertices that form a contour of the inside label, ordered CCW.
[...] # rest of planes
'''


class ConnectedComponent:
    def __init__(self, parent_cc_index, label, vertices_indices):
        self.parent_cc_index = parent_cc_index
        self.label = label
        self.vertices_indices = np.array(vertices_indices)

    @classmethod
    def from_cls(cls, csl_file):
        component = iter(next(csl_file).strip().split(" "))
        sizes = next(component).split("h") + [-1]

        # parent_cc_index is the index of the ConnectedComponent in which the hole lies (applies only for holes)
        n_vertices_in_component, parent_cc_index = int(sizes[0]), int(sizes[1])
        component = map(int, component)
        label = next(component)

        # ccw for non holes and cw for holes
        vertices_indices = list(component)
        assert len(vertices_indices) == n_vertices_in_component
        return cls(parent_cc_index, label, vertices_indices)

    def __len__(self):
        return len(self.vertices_indices)

    def __repr__(self):
        return f"{len(self.vertices_indices)}{'h'+str(self.parent_cc_index) if self.is_hole else ''} {self.label} {' '.join(map(str, self.vertices_indices))}\n"

    @property
    def is_hole(self):
        return self.parent_cc_index >= 0

    @property
    def edges_indices(self):
        e1 = self.vertices_indices
        e2 = np.concatenate((self.vertices_indices[1:], self.vertices_indices[0:1]))
        return np.stack((e1, e2)).T

class Plane:
    def __init__(self, plane_id: int, plane_params: tuple, vertices: np.array, connected_components: list, csl):
        assert len(plane_params) == 4

        self.csl = csl
        self.plane_id = plane_id

        self.vertices = vertices  # should be on the plane
        self.connected_components = connected_components

        # self.plane_params = plane_params  # Ax+By+Cz+D=0
        self.plane_params = plane_params
        self.normal = np.array(plane_params[0:3])
        self.normal /= np.linalg.norm(self.normal)

        self.plane_origin = plane_origin_from_params(plane_params)
        assert not self.is_empty

    def __repr__(self):
        plane = f"{self.plane_id} {len(self.vertices)} {len(self.connected_components)} {'{:.10f} {:.10f} {:.10f} {:.10f}'.format(*self.plane_params)}\n\n"
        verts = ''.join(['{:.10f} {:.10f} {:.10f}'.format(*vert) + '\n' for vert in self.vertices]) + "\n"
        ccs = ''.join(map(repr, self.connected_components))
        return plane + verts + ccs

    def __len__(self):
        return sum((len(cc) for cc in self.connected_components))

    def __isub__(self, point: np.array):
        assert point.shape == (3,)
        self.vertices -= point
        self.plane_origin -= point

        new_D = - np.dot(self.plane_params[:3], self.plane_origin)
        self.plane_params = self.plane_params[:3] + (new_D,)

    def __itruediv__(self, scale: float):
        self.vertices /= scale
        self.plane_origin /= scale

        new_D = - np.dot(self.plane_params[:3], self.plane_origin)
        self.plane_params = self.plane_params[:3] + (new_D,)

    def __imatmul__(self, rotation: PCA):
        if len(self.vertices) > 0:
            self.vertices = rotation.transform(self.vertices)
        self.plane_origin = rotation.transform([self.plane_origin])[0]
        self.normal = rotation.transform([self.normal])[0]

        new_D = - np.dot(self.plane_params[:3], self.plane_origin)
        self.plane_params = self.plane_params[:3] + (new_D,)

    @classmethod
    def from_csl_file(cls, csl_file, csl):
        line = next(csl_file).strip()
        plane_id, n_vertices, n_connected_components, a, b, c, d = \
            parse("{:d} {:d} {:d} {:f} {:f} {:f} {:f}", line)
        plane_params = (a, b, c, d)
        vertices = np.array([parse("{:f} {:f} {:f}", next(csl_file).strip()).fixed for i in range(n_vertices)])
        if n_vertices == 0:
            vertices = np.empty(shape=(0, 3))
        assert len(vertices) == n_vertices
        connected_components = [ConnectedComponent.from_cls(csl_file) for _ in range(n_connected_components)]
        return cls(plane_id, plane_params, vertices, connected_components, csl)

    @property
    def is_empty(self):
        return len(self.vertices) == 0

    @property
    def pca_projection(self):
        if self.is_empty:
            raise Exception("rotating empty plane")

        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(self.vertices)
        return pca.transform(self.vertices), pca

    @property
    def edges(self):
        edges = np.empty((0, 2), dtype=int)
        for cc in self.connected_components:
            edges = np.concatenate((edges, cc.edges_indices))
        return edges

class CSL:
    def __init__(self, model_name, plane_gen, n_labels):
        self.model_name = model_name
        self.n_labels = n_labels
        self.planes = plane_gen(self)

    def __len__(self):
        return sum((len(plane) for plane in self.planes))

    def __repr__(self):
        non_empty_planes = list(filter(lambda plane: len(plane.vertices) > 0, self.planes))
        return f"CSLC\n{len(non_empty_planes)} {self.n_labels} \n\n" + ''.join(map(repr, non_empty_planes))

    @classmethod
    def from_csl_file(cls, filename):
        model_name = path.basename(filename).split('.')[0]
        assert path.basename(filename).split('.')[-1].lower() == 'csl'
        with open(filename, 'r') as csl_file:
            csl_file = map(str.strip, filter(None, (line.rstrip() for line in csl_file)))
            assert next(csl_file).strip() == "CSLC"
            n_planes, n_labels = parse("{:d} {:d}", next(csl_file).strip())
            plane_gen = lambda csl: [Plane.from_csl_file(csl_file, csl) for _ in range(n_planes)]
            return cls(model_name, plane_gen, n_labels)

    @property
    def all_vertices(self):
        ver_list = (plane.vertices for plane in self.planes if not plane.is_empty)
        return np.array(list(chain(*ver_list)))

    @property
    def edges_verts(self):
        scene_verts = np.empty((0, 3))
        scene_edges = np.empty((0, 2), dtype=np.int32)
        for plane in self.planes:
            plane_vert_start = len(scene_verts)

            scene_verts = np.concatenate((scene_verts, plane.vertices))
            scene_edges = np.concatenate((scene_edges, plane.edges + plane_vert_start))

        return scene_edges, scene_verts

    def to_file(self, path):
        with open(path, 'w') as f:
            f.write(repr(self))

    def get_datasets(self, pool, n_refinements):

        return [pool.apply_async(SlicesDataset.from_csl, (self, i))
                for i in range(n_refinements)]

    def to_ply(self, file_name):
        edges, verts = self.edges_verts

        header = f'ply\n' \
                 f'format ascii 1.0\n' \
                 f'element vertex {len(verts)}\n' \
                 f'property float x\nproperty float y\nproperty float z\n' \
                 f'element edge {len(edges)}\n' \
                 f'property int vertex1\n' \
                 f'property int vertex2\n' \
                 f'property uchar red\n' \
                 f'property uchar green\n' \
                 f'property uchar blue\n' \
                 f'end_header\n'


        with open(file_name, 'w') as f:
            f.write(header)
            for v in verts:
                f.write('{:.10f} {:.10f} {:.10f}\n'.format(*v))
            for e in edges:
                f.write('{:d} {:d} 0 255 0\n'.format(*e))
