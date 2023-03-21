# OReX 
[![arXiv](https://img.shields.io/badge/arXiv-<2211.12886>-<COLOR>.svg)](https://arxiv.org/abs/2211.12886)

The official PyTorch implementation of the paper [**"OReX: Object Reconstruction from Planar Cross-sections Using Neural Fields"**](https://arxiv.org/abs/2211.12886).

[//]: # (![alt text]&#40;https://github.com/haimsaw/CrossSections/blob/Release/teaser.png?raw=true&#41;)
![alt text](https://github.com/haimsaw/OReX/blob/master/teaser.png?raw=true)



## Installation

Installation includes cloning the repo, creating virtual env and installing dependencies (dataset is already included).

```bash
git clone https://github.com/haimsaw/OReX.git
cd OReX

python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Reconstruct

Use the following command to run the code:

```bash
python3 Main.py ./output/directory ./path/to/input.csl --cuda_device dev_id 
```

For example this command will reconstruct the eight_15 model and save results to ./Artifacts:

```bash
python3 Main.py ./Artifacts ./Data/csl_with_ref/eight_15.csl --cuda_device 0 
```

See code for Hyperparams and additional args

## Dataset
All slices appear in the paper are under Data folder along with ref meshes (if exists) and slices in ply format.
Our code expects to receive input in the csl format (see below).

### Generating a .csl file

The following command will generate a csl file from a mesh by slicing it with random slices. csl and the reference mesh are saved to ./output/directory
```bash
python3 Slicer.py ./path/to/mesh.obj ./output/directory num_of_slices
```

### CSL file format

Our code expect to receive as input a .csl file whose format is as follows (comments are not allowed in actual files): 
```
CSLC # header
15 2  # number of planes, number of labels (should be at least 2 - inside and outside)

1 78 1 0.0 0.0 1.0 -0.86 # plane index (1-indexing, please state planes in order), number of vertices in the plane image (a hole is counted as another component), number of connected components, plane parameters A,B,C,D, such that Ax+By+Cz+D=0

0.10 0.08 0.86 # The vertices in x,y,z coordinates, should be on the plane.
0.09 0.08 0.86
0.08 0.09 0.86
0.07 0.09 0.86
[...] # rest of vertices

78 1 0 1 2 3 4 5 6 7 8 9 10 11 [...]  # image component: starts with the number of vertices, then label of the component (in case of a hole, h should be added and the index of the component contains the hole), then the indices of vertices that form a contour of the inside label, ordered CCW.
[...] # rest of components

[...] # rest of planes
```

## Acknowledgments
This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:

[ReStyle](https://github.com/yuval-alaluf/restyle-encoder), [MC-DC](https://github.com/BorisTheBrave/mc-dc), [NeRF](https://github.com/yenchenlin/nerf-pytorch)

## Bibtex
If you find this code useful in your research, please cite:

```
@article{sawdayee2023orex,
  title={OReX: Object Reconstruction from Planar Cross-sections Using Neural Fields},
  author={Sawdayee, Haim and Vaxman, Amir and Bermano, Amit H.},
  journal={arXiv preprint arXiv:2211.12886},
  year={2023}
}
```