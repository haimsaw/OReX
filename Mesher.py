from itertools import product
from multiprocessing import Pool
from os import path

from matplotlib import pyplot as plt

from Globals import n_process, output_path, StatsMgr, args
from Dataset.Helpers import *
from mcdc.common import adapt
from mcdc.dual_contour_3d import dual_contour_3d


def _dual_contouring(net_manager, meshing_resolution, label):

    print(f'Extracting mesh: res={meshing_resolution}')

    meshing_resolution_arr = np.array([meshing_resolution] * 3)

    # since in dc our vertices are inside the grid cells we need to have res+1 grid points
    x = np.linspace(-1.0, 1.0, meshing_resolution+1)
    y = np.linspace(-1.0, 1.0, meshing_resolution+1)
    z = np.linspace(-1.0, 1.0, meshing_resolution+1)
    xyzs = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape((-1, 3))


    labels = net_manager.get_predictions(xyzs).reshape(meshing_resolution_arr + 1)
    StatsMgr.setitem(f'dc_{label}_labels', (labels.min(), labels.max()))

    if labels.min() > 0 or labels.max() < 0:
        raise Exception('No zero level set')

    # dual_contour_3d uses grid points as coordinates
    # so i j k are the indices for the label (and not the actual point)
    ijk_to_xyz = Get_ijk_to_xyz(meshing_resolution_arr)
    f = GetFun(labels)
    f_normal = GetFunNormal(net_manager, ijk_to_xyz)

    _save_vert_placement(f, meshing_resolution, f'dc_{label}_vert_placement.png')

    mesh = dual_contour_3d(f, f_normal, ijk_to_xyz, *meshing_resolution_arr)
    mesh.to_tri()
    mesh.save(path.join(output_path,f'{"checkpoints" if label != "last" else "" }', 'mesh_{label}_{meshing_resolution}.obj'))

    return mesh


class GetFun:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, i, j, k):
        return self.labels[i][j][k]


class Get_ijk_to_xyz:
    def __init__(self, meshing_resolution):
        self.meshing_resolution = meshing_resolution

    def __call__(self, ijks):
        return 2 * ijks / self.meshing_resolution - 1


class GetFunNormal:
    def __init__(self, net_manager, ijk_to_xyz):
        self.net_manager = net_manager
        self.ijk_to_xyz = ijk_to_xyz
        self.ijks_to_grad = None

    def calc_grads(self, ijks_for_grad):
        if len(ijks_for_grad) > 0:
            # translate from ijk (index) coordinate system to xyz
            # where xyz = np.linspace(-1, 1, sampling_resolution_3d[i]+1, endpoint=True)
            ijks_for_grad = np.array(ijks_for_grad)
            xyzs_for_grad = self.ijk_to_xyz(ijks_for_grad)

            grads = self.net_manager.grad_wrt_input(xyzs_for_grad)

            self.ijks_to_grad = dict(zip(map(tuple, ijks_for_grad), grads))

        # delete this for pikiling
        self.net_manager = None
        self.ijk_to_xyz = None

    def __call__(self, i, j, k):
        if self.ijks_to_grad is not None:
            return self.ijks_to_grad[(i, j, k)]
        else:
            return np.array([0.0, 0.0, 0.0])


def _save_vert_placement(f, resolution, file_name):
    with Pool(n_process) as pool:
        inputs = product([f], range(resolution), range(resolution), range(resolution))
        all_changes_ratios = pool.starmap(_get_changes_vals, inputs)

    abs_ratios = [abs(ratio) for ratios in all_changes_ratios for ratio in ratios]

    plt.hist(abs_ratios, density=True, bins=100, range=(0, 1))
    plt.savefig(output_path + file_name, dpi=500)
    plt.close()
    plt.cla()
    plt.clf()


def _get_changes_vals(f, x, y, z):
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


def handle_meshes(trainer, label):
    if label == 'last':
        print(f'Starting finale meshing...')

        with StatsMgr.timer('last_meshing'):
            return _dual_contouring(trainer, args.meshing_resolution, label)

    else:
        with StatsMgr.timer('meshing', label):
            return _dual_contouring(trainer, args.meshing_resolution // 2, label)
