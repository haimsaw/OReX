"""Contains utilities common to 3d meshing methods"""

import math
import numpy as np


class V3:
    """A vector in 3D space"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def normalize(self):
        d = math.sqrt(self.x*self.x+self.y*self.y+self.z*self.z)
        return V3(self.x / d, self.y / d, self.z / d)


class Tri:
    """A 3d triangle"""
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def map(self, f):
        return Tri(f(self.v1), f(self.v2), f(self.v3))

    def __getitem__(self, idx):
        return [self.v1, self.v2, self.v3][idx]

class Quad:
    """A 3d quadrilateral (polygon with 4 vertices)"""
    def __init__(self, v1, v2, v3, v4):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.v4 = v4

    def map(self, f):
        return Quad(f(self.v1), f(self.v2), f(self.v3), f(self.v4))

    def swap(self, swap=True):
        if swap:
            return Quad(self.v4, self.v3, self.v2, self.v1)
        else:
            return Quad(self.v1, self.v2, self.v3, self.v4)

    def __getitem__(self, idx):
        return [self.v1, self.v2, self.v3, self.v4][idx]


class Mesh:
    """A collection of vertices, and faces between those vertices."""
    def __init__(self, verts=None, faces=None):
        self.verts = np.array(verts)
        self.faces = faces or []

    @property
    def vectors(self):
        return [self.verts[[face.v1-1, face.v2-1, face.v3-1, face.v4-1]] for face in self.faces]

    def extend(self, other):
        l = len(self.verts)
        f = lambda v: v + l
        self.verts.extend(other.verts)
        self.faces.extend(face.map(f) for face in other.faces)

    def __add__(self, other):
        r = Mesh()
        r.extend(self)
        r.extend(other)
        return r

    def translate(self, offset):
        new_verts = [V3(v.x + offset.x, v.y + offset.y, v.z + offset.z) for v in self.verts]
        return Mesh(new_verts, self.faces)

    def to_tri(self):
        tri_faces = []
        for face in self.faces:
            if isinstance(face, Tri):
                tri_faces.append(face)
            elif isinstance(face, Quad):
                tri_faces.append(Tri(face.v1, face.v2, face.v3))
                tri_faces.append(Tri(face.v3, face.v4, face.v1))
        self.faces = tri_faces

    def save(self, path):
        with open(path, "w") as f:
            make_obj(f, self)


def make_obj(f, mesh):
    """Crude export to Wavefront mesh format"""
    for v in mesh.verts:
        f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
    for face in mesh.faces:
        if isinstance(face, Quad):
            f.write("f {} {} {} {}\n".format(face.v4 + 1, face.v3 + 1, face.v2 + 1, face.v1 + 1))
        if isinstance(face, Tri):
            f.write("f {} {} {}\n".format(face.v3 + 1, face.v2 + 1, face.v1 + 1))
