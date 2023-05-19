# Medusa installation

Medusa is a Python package which works with Python version 3.9 and on Linux and and Mac (except Mac with M1/M2 chips). Most of Medusa's functionality will in fact also work on Windows, with the exception of rendering (as [pytorch3d](https://pytorch3d.org/) cannot be automatically installed on Windows).

We strongly recommend to install the `medusa` package in a separate Python environment, using for example [conda](https://anaconda.org/anaconda/conda). If you'd use *conda*, you can create a new environment named "medusa" with Python 3.9 as follows:

```console
conda create -n medusa python=3.9
```

Then, to activate the environment, run:

```console
conda activate medusa
```

The next step is to install Medusa. Medusa actually offers two version of the package:
`medusa` and `medusa-gpu`, where the latter can be used instead of the former if you
have access to an NVIDIA GPU; apart from the installation, both versions contain exactly
the same functionality. Actually, `medusa-gpu` can also be installed and used on systems
without a GPU, but the installation is noticeably larger (~2GB, instead of 300MB for the
CPU version). When you're not sure whether you have access to an appropriate GPU, we
recommend installing the regular (cpu) `medusa` package.

To install Medusa, run one of the commands listed below in your terminal (with the right
environment activated):

`````{tab-set}

````{tab-item} medusa (CPU)
```console
pip install https://github.com/medusa-4D/medusa/releases/download/v0.0.5/medusa-0.0.5-py3-none-any.whl
```
````

````{tab-item} medusa-gpu
```console
pip install https://github.com/medusa-4D/medusa/releases/download/v0.0.5/medusa_gpu-0.0.5-py3-none-any.whl
```
````

`````

```{note}
While installing Python packages/wheels from other locations than PyPI is generally
discouraged, Medusa actually hosts its builds in its own Github repository (as you can
see in the install commands above). The reason for doing so (instead of on PyPI) is that
Medusa depends on a specific version of [PyTorch](https://pytorch.org/), which itself
is not available on PyPI (only as a wheel). Listing non-PyPI dependencies in packages
is not permitted by PyPI, which is why Medusa wheels are hosted on Github.

If you want to build Medusa yourself, you can clone the repository and run the
`build_wheels` script, which will create a directory `dist` with two wheel files
(one for `medusa` and one for `medusa-gpu`).
```

## Downloading external data

At this point, `medusa` can be used, but only the Mediapipe reconstruction model will be
available. To be able to use the FLAME-based {cite}`li2017learning` reconstruction models such as
DECA {cite}`feng2021learning`, and EMOCA {cite}`danvevcek2022emoca`,
you need to download some additional data. Importantly, before you do, you need to
[register](https://flame.is.tue.mpg.de/register.php) on the [FLAME website](https://flame.is.tue.mpg.de/index.html)
and accept their [license terms](https://flame.is.tue.mpg.de/modellicense.html).

After creating an account, you can download all external data with the
`medusa_download_ext_data` command. To download all data to new directory
(default location: `~/.medusa_ext_data`), you'd run:

```console
medusa_download_ext_data --directory medusa_ext_data --username your_flame_username --password your_flame_passwd
```

where `your_flame_username` and `your_flame_passwd` are the username and password associated
with the account you created on the FLAME website. After all data has been downloaded
(~1.8GB), all Medusa functionality should be available!

## Installation from source

If you want to install Medusa from source, check out the [for developers](../misc/for_developers) page.
