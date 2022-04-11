# GMFX

GMFX is a Python toolbox to analyze and build 3D Generative Models of Facial eXpressions. It uses a deep learning-based model ([EMOCA](https://emoca.is.tue.mpg.de/)) to reconstruct 3D faces from images or videos and provides tools to subsequently preprocess and analyze the 3D meshes.

## Installation

We strongly recommend to install the `gmfx` package in a separate [conda environment](https://anaconda.org/anaconda/conda). Assuming you have access to the `conda` (by installing [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), run the following in your terminal to create a new environment named `gmfx` with Python version 3.9:

```
conda create -n gmfx python=3.9
```

Then, activate the environment and install `pytorch` (used by the 3D reconstruction model) as follows:

```
conda activate gmfx
conda install pytorch cudatoolkit=11.3 -c pytorch
```

The next step is to download the `gmfx` package from Github, either as [zip file](https://github.com/lukassnoek/gmfx/archive/refs/heads/master.zip) (which you need to extract afterwards) or using `git` (i.e., `git clone https://github.com/lukassnoek/gmfx.git`). The first thing to do, before installing `gmfx` intself, is to compile and install the C++ based rasterizer (to render 3D shapes as images) it contains. To install the rasterizer, you can run the following command (assuming that you're in the root of the downloaded repository):

```
cd gmfx/render/rasterizer && pip install . && cd ../../..
```

Finally, again assuming you're in the root of the downloaded repository (and you have the `gmfx` conda environment activated), run the following command to install `gmfx` and its dependencies:

```
pip install .
```

At this point, the package's CLI tools (e.g., `gmfx_recon`) and Python API should be available. To verify, run the following commands in your terminal:

```
gmfx_recon --help  # to verify the CLI (should print out options)
python -c 'import gmfx'  # to verify the Python API (shouldn't error)
```

## Download DECA

If you are on one of Glasgow's "deepnet" servers and have access to `Project0294` (if not, ask Oliver), you can simply copy the DECA model and associated files as follows (assuming you're currently in the root of the `gmfx` package):

```
cp /analyse/Project0294/gmfx_data/data/* gmfx/recon/deca/data/
```

If not, you should follow the download instructions in the [DECA Github repository](https://github.com/YadiraF/DECA) under "Getting started / Usage / Prepare data". Instead of unzipping it in `./data` (as written in the instructions), place the files in `gmfx/recon/deca/data`.

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
