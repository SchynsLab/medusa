# Medusa installation

Medusa is a Python package which works with Python version 3.9 and on Linux and and Mac (x86_64 architectures only). Most of Medusa's functionality will in fact also work on Windows and M1/M2 Macs (arm64 architecture), with the exception of rendering (as [pytorch3d](https://pytorch3d.org/) cannot be automatically installed on Windows and Mac M1/M2).

We strongly recommend to install the `medusa` package in a separate Python environment, using for example [conda](https://anaconda.org/anaconda/conda). If you'd use *conda*, you can create a new environment named "medusa" with Python 3.9 as follows:

```console
conda create -n medusa python=3.9
```

Then, to activate the environment, run:

```console
conda activate medusa
```

The next step is to install Medusa. Medusa's install will be relatively large (~2GB) as
it will also install PyTorch with CUDA support, even if your system does not have access
to a GPU (which makes automatic installation a lot easier); Medusa will make sure set
the default 'device' to 'cpu' so PyTorch will work as expected.

Medusa can be installed using `pip` as follows:

```console
pip install https://github.com/SchynsLab/medusa/releases/download/v0.0.6/medusa-0.0.6-py3-none-any.whl
```

```{note}
While installing Python packages/wheels from other locations than PyPI is generally
discouraged, Medusa actually hosts its builds in its own Github repository (as you can
see in the install commands above). The reason for doing so (instead of on PyPI) is that
Medusa depends on a specific version of [PyTorch](https://pytorch.org/), which itself
is not available on PyPI (only as a wheel). Listing non-PyPI dependencies in packages
is not permitted by PyPI, which is why Medusa wheels are hosted on Github.

If you want to build Medusa yourself, you can clone the repository and run the
`build_wheel` script, which will create a directory `dist` with a wheel file that can
then be installed using `pip` as usual.
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
