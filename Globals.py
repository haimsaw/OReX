import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from os import cpu_count, path
from time import time

import torch

parser = argparse.ArgumentParser(description='Run OReX.')

# run
parser.add_argument('out_dir', type=str, help='out directory to save outputs')
parser.add_argument('input', type=str, default='armadillo', help='path to input csl')

parser.add_argument('--cuda_device', type=int, default=-1, help='which gpu device to use')
parser.add_argument('--seed', type=int, default=11, help='random seed to use (-1 for random)')

parser.add_argument('-centralize_csl', action='store_true')

# Sampling
parser.add_argument('--bounding_margin', type=float, default=0.05, help='the margin of the convex hull')

parser.add_argument('--n_white_noise', type=int, default=2048, help='n of random points to sample at each plane')
parser.add_argument('--n_samples', nargs='*', type=int, default=[2, 2, 3, 3, 4, 5],
                    help='samples around edges for each refinement level')
parser.add_argument('--sampling_radius_exp', nargs='*', type=int, default=[4, 5, 5, 6, 6, 7],
                    help='the exp of sampling radius (base 0.5) at each refinement level')
parser.add_argument('--n_samples_boundary', type=int, default=2 ** 14)

parser.add_argument('--meshing_resolution', type=int, default=300, help='meshing Sampling res')

# architecture
parser.add_argument('--embedding_freqs', type=int, default=5, help='number of embedding freqs')

parser.add_argument('--hidden_state_size', type=int, default=32, help='hidden state size')

parser.add_argument('--hidden_layers_width', type=int, default=7, help='n of hidden layers')
parser.add_argument('--hidden_layers_height', type=int, default=64, help='n of nuruns in each layer')

parser.add_argument('--n_iterations', type=int, default=10, help='n of times we iterate OReX net')

# loss
parser.add_argument('--eikonal_lambda', type=float, default=0, help='Eikonal to loss')

parser.add_argument('--eikonal_hinge_lambda', type=float, default=1e-4, help='Eikonal to loss')
parser.add_argument('--hinge_alpha', type=float, default=100, help='Eikonal to loss')

# training
parser.add_argument('--lr', type=float, default=1e-2, help='initial lr')
parser.add_argument('--scheduler_gamma', type=float, default=0.9, help='exponential lr decay')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='l2 regularization')

parser.add_argument('--batch_size_exp', type=int, default=14, help='the exponent of the batch size (base 2)')
parser.add_argument('--scheduler_step', type=int, default=10, help='in how many iterations should we reduce lr')
parser.add_argument('--epochs_batches', nargs='*', type=int, default=[50, 50, 100, 100, 150, 200],
                    help='number of epochs in each refinement level')

parser.add_argument('--n_used_datasets', type=int, default=3, help='n of datasets used during at each epoch batch')

if 'Main.py' in sys.argv[0]:
    args = parser.parse_args()
else:
    args = parser.parse_args(['null']*2)

assert len(args.epochs_batches) <= len(args.sampling_radius_exp) and len(args.epochs_batches) <= len(args.n_samples)


class StatsMgr:
    _stats = {'input_file': args.input,
              'timestamp': str(datetime.now()),
              'timings': {},
              'dataset_size': {}
              }

    @staticmethod
    @contextmanager
    def timer(section, i=None):
        ts = time()
        try:
            yield None

        finally:
            elapsed = time() - ts
            if i is None:
                StatsMgr._stats['timings'][section] = elapsed
            else:
                try:
                    StatsMgr._stats['timings'][section][i] = elapsed
                except KeyError:
                    StatsMgr._stats['timings'][section] = {i: elapsed}

    @staticmethod
    def setitem(key, value, i=None):
        if i is None:
            StatsMgr._stats[key] = value
        else:
            try:
                StatsMgr._stats[key][i] = value
            except KeyError:
                StatsMgr._stats[key] = {i: value}

    def __class_getitem__(cls, item):
        return cls._stats[item]

    @staticmethod
    def get_str():
        return json.dumps(StatsMgr._stats, indent=4)


model_name = path.split(args.input)[1].split('.')[0]
output_path = path.join(args.out_dir, model_name, '')
n_process = min(5, cpu_count() // 2)
device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
INSIDE_LABEL = 0.0
OUTSIDE_LABEL = 1.0
