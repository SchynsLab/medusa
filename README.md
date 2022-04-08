# GMFX
A Python implementation of a Generative Model of Facial eXpressions, leveraging a deep learning-based
single-image 3D reconstruction model ([DECA](https://deca.is.tue.mpg.de/)).

## Installation
The installation of `gmfx` is not trivial because it needs a specific version of `pytorch` (used by DECA and the face detection model), which itself depends on the specific CUDA version installed on your platform.

It's probably best to use a custom environment to install `gmfx`. To do so using Anaconda, run the following command:

```
conda create -n gmfx python=3.9
```

Then, install `pytorch` and dependencies as follows:

```
conda create -n gmfx python=3.9
conda activate gmfx
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
```

Then, we need to install the CUDA-based rasterizer shipped with DECA. To do so, run the following:

```
cd gmfx/render/rasterizer && pip install . && cd ../../..
```

Finally, install the `gmfx` package itself (note the `.` at the end):

```
pip install .
```

## Download DECA
If you are on one of Glasgow's "deepnet" servers and have access to `Project0294` (if not, ask Oliver), you can simply copy the DECA model and associated files as follows (assuming you're currently in the root of the `gmfx` package):

```
cp /analyse/Project0294/gmfx_data/data gmfx/recon/deca/
```

If not, you should download DECA [here](https://deca.is.tue.mpg.de/) (you need to register and accept their license first) and unpack the ZIP file in `gmfx/recon/deca/data`.

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

If you're on one of the "deepnet" servers, you can copy some test data from `Project0294`:

```
cp -r /analyse/Project0294/gmfx_data/test_data .
```

And then, to test the reconstruction, run:

```
gmfx_recon test_data/task-browraiser --participant-label sub-01 --device cuda
```

This will create a default output directory at `test_data/task-browraiser/derivatives/sub-01` with the reconstruction data in an HDF5 file (ending in `.h5`) and a video of the reconstruction in "world space" (in the space of the cropped image) another one in "image space" (projected back into the original image space).

## Using the Python interface

...
