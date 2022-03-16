# GMFX
An Python implementation of a Generative Model of Facial eXpressions, leveraging a deep learning-based
single-image 3D reconstruction model ([DECA](https://deca.is.tue.mpg.de/)).

## Installation
The installation of `gmfx` is not trivial because it needs a specific version of `pytorch` (used by DECA and the face detection model), which itself depends on the specific CUDA version installed on your platform.

It's probably best to use a custom environment to install `gmfx`. To do so using Anaconda, run the following command:

```
conda create -n gmfx python=3.9
```

The next step is to install `pytorch`. Its installation is specific to your platform, as it depends on the installed CUDA version. For users on the Glasgow "deepnet" servers, you should run the following command:

```
bash install_pytorch.sh
```

Then, we need to install the CUDA-based rasterizer shipped with DECA. To do so, run the following:

```
cd gmfx/render/rasterizer && pip install . && cd ../../..
```

Finally, install the `gmfx` package itself (note the `.` at the end):

```
pip install .
```

## Using the command line interface (CLI)
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

* `gmfx_recon` (for reconstruction from a 2D image to 3D mesh)
* `gmfx_align` (for spatially aligning all meshes across time)
* `gmfx_interpolate` (to make such as mesh is equidistant in time)
* `gmfx_filter` (to high- and lowpass each vertex across time)

To see the mandatory and optional arguments to these CLI programs, run the command with the `--help` flag, e.g.:

```
gmfx_recon --help
```

The folder `gmfx/test_data` contains some example data to try out the package.

## Using the Python interface

...