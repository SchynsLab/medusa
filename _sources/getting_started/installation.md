# Installation

We strongly recommend to install the `medusa` package in a separate [conda environment](https://anaconda.org/anaconda/conda). Assuming you have access to the `conda` (by installing [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), run the following in your terminal to create a new environment named `medusa` with Python version 3.9:

```
conda create -n medusa python=3.9
```

Then, activate the environment and install `pytorch` (used by the 3D reconstruction model) as follows:

```
conda activate medusa
conda install pytorch cudatoolkit=11.3 -c pytorch
```

The next step is to download the `medusa` package from Github, either as [zip file](https://github.com/lukassnoek/medusa/archive/refs/heads/master.zip) (which you need to extract afterwards) or using `git` (i.e., `git clone https://github.com/lukassnoek/medusa.git`). Finally, again assuming you're in the root of the downloaded repository (and you have the `medusa` conda environment activated), run the following command to install `medusa` and its dependencies:

```
pip install .
```

At this point, the package's CLI tools (e.g., `medusa_videorecon`) and Python API should be available. To verify, run the following commands in your terminal:

```
medusa_videorecon --help  # to verify the CLI (should print out options)
python -c 'import medusa'  # to verify the Python API (shouldn't error)
```

## Download DECA

If you are on one of Glasgow's "deepnet" servers and have access to `Project0294` (if not, ask Oliver), you can simply copy the EMOCA model and associated files as follows (assuming you're currently in the root of the `medusa` package):

```
cp /analyse/Project0294/medusa_data/data/* medusa/recon/deca/data/
```

If not ... TBD

## Using the command line interface (CLI)
The command line interface can be used to preprocess video data step by step. The first step, reconstruction of the video frames into 3D meshes, assumes the following directory structure:

```
data
└── sub-01
    ├── sub-01_task-mini_events.tsv
    ├── sub-01_task-mini_frametimes.tsv
    └── sub-01_task-mini.mp4
```

where `data` is the toplevel directory, which contains one or more subject-level directories (e.g., `sub-01`). Each subject directory contains at least a video file (for now, only `mp4` files are allowed) ending in `_video.mp4` and a "frame times" file ending in `_frametimes.tsv`. This tabular file (with tab separators) should contain a column named `t` which contains, for every frame in the video, the time in seconds of the acquisition of the frame. The file ending in `_events.tsv` is not strictly necessary for preprocessing, but is needed for the analysis phase.