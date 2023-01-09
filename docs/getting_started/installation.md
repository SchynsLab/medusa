# Medusa installation

Medusa is a Python package which works with Python versions 3.9 and above. We recommend
using Python version 3.9. Moreover, we strongly recommend to install the `medusa` package
in a separate environment, using for example [conda](https://anaconda.org/anaconda/conda).
If you'd use *conda*, you can create a new environment named "medusa" with python 3.9
as follows:

```console
conda create -n medusa python=3.9
```

Then, to activate the environment, run:

```console
conda activate medusa
```

The next step is to install Medusa. Medusa actually offers two version of the package:
`medusa` and `medusa-gpu`, where the latter can be used instead of the former if you
have access to an NVIDIA GPU (and CUDA version 11.6). When you're not sure whether
you have access to an appropriate GPU, install the regular `medusa` package.

`````{tab-set}

````{tab-item} medusa (CPU)
```console
pip install https://github.com/medusa-4D/medusa/releases/download/v0.0.3/medusa-0.0.3-py3-none-any.whl
```
````

````{tab-item} medusa-gpu
```console
pip install https://github.com/medusa-4D/medusa/releases/download/v0.0.3/medusa_gpu-0.0.3-py3-none-any.whl
```
````

`````

At this point, `medusa` can be used, but only the Mediapipe reconstruction model can be
used. To be able to use the FLAME-based reconstruction models such as DECA, EMOCA, and
Spectre, you need to download some additional data. Importantly, before you do, you need
to [register](https://flame.is.tue.mpg.de/register.php) on the [FLAME website](https://flame.is.tue.mpg.de/index.html)
and accept their [license terms](https://flame.is.tue.mpg.de/modellicense.html).

After creating an account, you can download all external data with the
`medusa_download_ext_data` command. To download all data to new directory
(medusa_ext_data), you'd run:

```console
medusa_download_ext_data --directory medusa_ext_data --username your_flame_username --password your_flame_passwd
```

After all data has been downloaded (~1.8GB), all Medusa functionality should be available!
