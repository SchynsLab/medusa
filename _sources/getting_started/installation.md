# Medusa installation

Medusa is a Python package which works with Python versions 3.6 and above. We recommend
using Python version 3.9. Moreover, we strongly recommend to install the `medusa` package
in a separate [conda environment](https://anaconda.org/anaconda/conda). Assuming you have
access to the `conda` command (by installing [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), run the
following command in your terminal to create a new environment named `medusa` with Python
version 3.9:

```console
conda create -n medusa python=3.9
```

To activate the environment, run:

```console
conda activate medusa
```

Several reconstruction models depend on the `pytorch` Python package. Also, some
models (like EMOCA and FAN) can be run on the GPU (instead of CPU), which offers
substantial decreases in runtime. If you have access to a GPU, you can enable
GPU processing by installing the appropriate CUDA toolkit as well. Run the command
below depending on whether you want to run the models on GPU or CPU only:

````{tabbed} GPU
```
conda install pytorch cudatoolkit=11.3 -c pytorch
```
````

````{tabbed} CPU
```
conda install pytorch cpuonly -c pytorch
```
````

For more detailed information on how to install `pytorch` on different platforms
(e.g., Mac), check out the `pytorch` [website](https://pytorch.org/).

The next step is to download the `medusa` package from Github, either as [zip file](https://github.com/lukassnoek/medusa/archive/refs/heads/master.zip) (which you need to extract afterwards) or using `git` (i.e., `git clone https://github.com/lukassnoek/medusa.git`). Finally, again assuming you're in the root of the downloaded repository (and you have the `medusa` conda environment activated), run the following command to install `medusa` and its dependencies:

```console
pip install .
```

At this point, the package's CLI tools (e.g., `medusa_videorecon`) and Python API should be available. To verify, run the following commands in your terminal:

```
medusa_videorecon --help  # to verify the CLI (should print out options)
python -c 'import medusa'  # to verify the Python API (shouldn't error)
```

## Additional reconstruction models

The [Mediapipe](https://google.github.io/mediapipe/solutions/face_mesh) model works out-of-the-box, but Medusa can work with any reconstruction model as long as it takes as input an image and outputs a set of vertices.

### FLAME-based models

We also support reconstruction using [FLAME](https://flame.is.tue.mpg.de/)-based
models, such as [DECA](https://deca.is.tue.mpg.de/) and [EMOCA](https://emoca.is.tue.mpg.de/). These models are not
part of the core Medusa package, but are implemented in a [separate package](https://github.com/medusa-4D/flame),
`flame`. The reason for doing so is that the implementation of FLAME-based models is relatively complicated and
also needs additional data to work properly. We describe below which data you need and where to get it
and how to install the `flame` package.

First of all, download the `flame` package here: [https://github.com/medusa-4D/flame](https://github.com/medusa-4D/flame).
No need to install it yet; first, you need to download some data.

If you want to use all functionality from the `flame` package, you need to download three files:
`FLAME2020.zip` (the FLAME 3D morphable model files), `deca_model.tar` (the DECA reconstruction model weights),
and `EMOCA.zip` (the EMOCA reconstruction model weights). 

The FLAME model is necessary to convert the shape predictions from DECA and EMOCA into a dense
mesh. You can download the model [here](https://flame.is.tue.mpg.de/download.php). You
need to register an account before being able to do so. After logging in, download
the "FLAME 2020 (fixed mouth, improved expressions, more data)" zip file (`FLAME2020.zip`).

The EMOCA model, i.e., the model that generates predictions of FLAME components from images,
can be downloaded [here](https://emoca.is.tue.mpg.de/download.php). Again, you need to register an account
before being able to do so. After logging in, download the "EMOCA" file (`EMOCA.zip`)
lister under the "Face Reconstruction" header. 

Finally, the DECA model (`deca_model.tar`) can be downloaded [here](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view).

The next step is downloading the `flame` package itself: [https://github.com/medusa-4D/flame](https://github.com/medusa-4D/flame).
Make sure all three files (`FLAME2020.zip`, `deca_model.tar`, and `EMOCA.zip`) are placed in the same directory, for example,
`ext_data` in the root of the `flame` package. This directory should look as follows:

```
ext_data/
    ├── deca_model.tar
    ├── EMOCA.zip
    └── FLAME2020.zip
```

Finally, run the `validate_external_data.py` script, located in the root of the `flame` package.
It takes a single positional argument, which should point towards the directory with the downloaded data (e.g., `ext_data`)
If you plan on running the EMOCA model on CPU only (i.e., not on GPU),
make sure to add `--cpu` to the command below.

```console
python validate_external_data.py ./ext_data
```

This script will unpack and reorganize the FLAME and EMOCA data such that it can be used
in `flame`. After running the script, your directory with datashould look like this:

```
ext_data/
    ├── deca_model.tar
    ├── FLAME/
    │   ├── Readme.pdf
    │   ├── female_model.pkl
    │   ├── generic_model.pkl
    │   └── male_model.pkl
    └── EMOCA/
        └── emoca.ckpt
```

Now, you can install the `flame` package by running the following command from the root of the package:

```console
pip install .
```
