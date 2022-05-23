# Installation

Medusa is a Python package which works with Python versions 3.6 and above. We recommend
using Python version 3.9. Moreover, we strongly recommend to install the `medusa` package
in a separate [conda environment](https://anaconda.org/anaconda/conda). Assuming you have
access to the `conda` command (by installing [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), run the
following command in your terminal to create a new environment named `medusa` with Python
version 3.9:

```
conda create -n medusa python=3.9
```

To activate the environment, run:

```
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

```
pip install .
```

At this point, the package's CLI tools (e.g., `medusa_videorecon`) and Python API should be available. To verify, run the following commands in your terminal:

```
medusa_videorecon --help  # to verify the CLI (should print out options)
python -c 'import medusa'  # to verify the Python API (shouldn't error)
```

## Download FLAME & EMOCA

The [Mediapipe](https://google.github.io/mediapipe/solutions/face_mesh) and
[FAN-3D](https://github.com/1adrianb/face-alignment) models work out-of-the-box, but
the EMOCA model needs additional data to work properly. In order to download this data,
you need to create an account and agree to their license. You need to download two 
files: the FLAME model and the EMOCA model.

The FLAME model is necessary to convert the shape predictions from EMOCA into a dense
mesh. You can download the model [here](https://flame.is.tue.mpg.de/download.php). You
need to register an account before being able to do so. After logging in, download
the "FLAME 2020 (fixed mouth, improved expressions, more data)" zip file. Then,
create a new directory named `ext_data` ("external data") in the root of the Medusa
repository and move the `FLAME2020.zip` file here (i.e., `ext_data/FLAME2020.zip`).

Second, you need to download the EMOCA model itself, i.e., the model that generates
predictions of FLAME components from images. You can download the model
[here](https://emoca.is.tue.mpg.de/download.php). Again, you need to register an account
before being able to do so. After logging in, download the "EMOCA" file (EMOCA.zip)
lister under the "Face Reconstruction" header. When the download is finished,
move the EMOCA.zip file in the `ext_data` folder (i.e, `ext_data/EMOCA.zip`). The 
`ext_data` folder should look like this now:

```
ext_data/
    ├── EMOCA.zip
    └── FLAME2020.zip
```

Finally, run the `validate_external_data.py` script, located in the root of the Medusa
repository. If you plan on running the EMOCA model on CPU only (i.e., not on GPU),
make sure to add `--cpu` to the command below.

```
python validate_external_data.py
```

This script will unpack and reorganize the FLAME and EMOCA data such that it can be used
in Medusa. After running the script, the `ext_data` folder should look like this:

```
ext_data/
    ├── FLAME/
    │   ├── Readme.pdf
    │   ├── female_model.pkl
    │   ├── generic_model.pkl
    │   └── male_model.pkl
    └── EMOCA/
        └── emoca.ckpt
```
