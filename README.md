# GMFX
An Python implementation of a Generative Model of Facial eXpressions, leveraging a deep learning-based
single-image 3D reconstruction model ([DECA](https://deca.is.tue.mpg.de/)).

## Installation
To use `pyface`, you need to both install DECA (the 3D mesh reconstruction model) and `pyface` itself.

### Downloading data

### Installing DECA
This packages uses [DECA](https://github.com/YadiraF/DECA) to reconstruct face meshes. To use `pyface`, you need to install the fork I created [here](https://github.com/lukassnoek/DECA), which is an installable version of the original (and with some bugfixes). Before you do so, I'd recommend create a new conda environment:

```
conda create -n deca python=3.8
```

And then then clone and install [fork](https://github.com/lukassnoek/DECA) as follows (from the root of the cloned directory):

```
bash install.sh
```

Note that this is only tested on Linux! YMMV on Mac or Windows. The script installs `pytorch` (1.9.0) and `pytorch3d` (0.5.0) as well as the `deca` package. To test the installation, you can try the following (from the root of the DECA directory)

```
python tests/test_installation.py
```

### Installing `pyface`
To install `pyface`, simply run the following (note the dot after `install`) from the root of the `pyface` directory:

```
pip install .
```

To test the installation, try the following:

```
pyface_recon tests/test.jpg --save-plot --device cuda
```

This runs the DECA model on an example image (`test.jpg`) and outputs an image of the reconstructed face mesh (`recon_test.jpg`). Use `--device cpu` if you don't have access to an NVIDIA GPU.

## Using the command line interface
The command line interface can be used to preprocess video data step by step. The first step, reconstruction of the video frames into 3D meshes, assumes the following directory structure:

```
data
└── sub-01
    ├── sub-01_task-mini_events.tsv
    ├── sub-01_task-mini_frametimes.tsv
    └── sub-01_task-mini_video.mp4
```

where `data` is the toplevel directory, which contains one or more subject-level directories (e.g., `sub-01`). Each subject directory contains at least a video file (for now, only `mp4` files are allowed) ending in `_video.mp4` and a "frame times" file ending in `_frametimes.tsv`. This tabular file (with tab separators) should contain a column named `t` which contains, for every frame in the video, the time in seconds of the acquisition of the frame. The file ending in `_events.tsv` is not strictly necessary for preprocessing, but is needed for the analysis phase.

The following preprocessing CLI programs are available, which should be run in the order outlined below:

* `pyface_recon` (for reconstruction from a 2D image to 3D mesh)
* `pyface_align` (for spatially aligning all meshes across time)
* `pyface_interpolate` (to make such as mesh is equidistant in time)
* `pyface_filter` (to high- and lowpass each vertex across time)

To see the mandatory and optional arguments to these CLI programs, run the command with the `--help` flag, e.g.,:

```
pyface_recon --help
```

The folder `pyface/data/test_data` contains some example data to try out the package.