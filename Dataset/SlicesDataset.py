import torch
from torch.utils.data import Dataset

from Dataset.Helpers import *
from Dataset.Sampler import sample_plane, sample_hull
from Globals import OUTSIDE_LABEL


class SlicesDataset(Dataset):
    def __init__(self, xyzs, labels, boundary_xyzs=None):

        if boundary_xyzs is None:
            boundary_xyzs = np.empty((0, 3))
        self.densities = torch.tensor(np.concatenate((labels, np.full(len(boundary_xyzs), OUTSIDE_LABEL)))).view((-1, 1))
        self.xyzs = torch.tensor(np.concatenate((xyzs, boundary_xyzs)))

        self.size = len(self.xyzs)

    @classmethod
    def from_csl(cls, csl, gen):

        data = [sample_plane(plane, gen) for plane in csl.planes]
        xyzs_list, labels_list = list(zip(*data))
        boundary = sample_hull(csl)

        return cls(np.concatenate(xyzs_list), np.concatenate(labels_list), boundary)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.xyzs[idx], self.densities[idx]

    def to_ply(self, file_name):
        header = f'ply\nformat ascii 1.0\nelement vertex {self.size}\n' \
                 f'property float x\nproperty float y\nproperty float z\n' \
                 f'property float quality\n' \
                 f'property uchar red\n' \
                 f'property uchar green\n' \
                 f'property uchar blue\n' \
                 f'element face 0\nproperty list uchar int vertex_index\nend_header\n'

        with open(file_name, 'w') as f:
            f.write(header)
            for xyz, density in zip(self.xyzs, self.densities):
                color = np.zeros(3, dtype=int)
                color[int(density * 2)] = 255
                f.write('{:.10f} {:.10f} {:.10f} {:.10f} {} {} {}\n'.format(*xyz, density[0], *color))
