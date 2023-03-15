# For developers

This page outlines some additional information for developers that intend to contribute
to the package.

## Installation from source

After cloning or downloading the source code from [Github](https://github.com/medusa-4D/medusa) and
assuming you have a working Python distribution (version==3.9), you need to install a couple
of dependencies before installing Medusa:

```console
pip install toml click poetry
```

Then, to determine whether the installer will install the CPU or GPU version of Medusa,
run the following command in the root of the Medusa repository:

```console
python set_package_version.py {cpu,gpu}
```

where you choose either "cpu" or "gpu". This will create a customized `pyproject.toml` file
with the correct PyTorch version.

Finally, install Medusa by running:

```console
poetry install
```

This may take a while (especially for the GPU version) as it will resolve and download
all dependencies. After the command has finished running, Medusa has been installed in
editable mode. This means that you can edit the code and changes will be effective
immediately without having to reinstall the package.

## Testing

Medusa contains an extensive set of unit tests in the *tests* directory (in the repository's
root). To execute all tests, run the following command in the repository root:

```console
./ci/run_unittests
```

which clears all previous test outputs and runs the entire test suite with coverage. To
just run *pytest* (the testing package that Medusa uses), run:

```console
pytest tests/
```

## Building documentation

To build the documentation (website including API docs), run the following in the root 
of the repository:

```console
./docs/build_docs
```

To just build the Python API documentation, run:

```console
sphinx-build -b html ./docs/autoapi ./docs/autoapi/_build
```

To just build the website without the API docs, run:

```console
jupyter-book build ./docs/
```

The website HTML code is deposited in `./docs/_build`.
