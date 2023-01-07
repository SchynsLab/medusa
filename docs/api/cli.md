# Command-line interface

If you're familiar with using terminals and command line interfaces (CLI), Medusa's
CLI might be useful for you. The CLI exposes Medusa's most important functionality
to the command line, which can be used after installing the Python package. Alternatively,
you can use the [Python API](./python).

All CLI commands start with `medusa_` followed by the operation that it exposes.
For example, `medusa_videorecon` is a command-line interface of the package's Python
function [`videorecon`](./python/recon/recon/index). The arguments and options of each CLI
command can be inspected by running the command with a single option `--help`. For
example:

```console
$ medusa_videorecon --help

Usage: medusa_videorecon [OPTIONS] VIDEO_PATH

  Performs frame-by-frame 3D face reconstruction of a video file.

Options:
  -o, --out PATH                  File to save output to (shouldn't have an
                                  extension)
  -r, --recon-model [spectre-coarse|emoca-dense|emoca-coarse|deca-dense|deca-coarse|mediapipe]
                                  Name of the reconstruction model
  --device [cpu|cuda]             Device to run the reconstruction on (only
                                  relevant for EMOCA
  -n, --n-frames INTEGER          Number of frames to process
  -b, --batch-size INTEGER        Batch size of inputs to recon model
  --help                          Show this message and exit.
```

For example, the `medusa_videorecon` command has a single mandatory argument,
`VIDEO_PATH`, and several (non-mandatory) options, like `--recon-model`.
If the option accepts an argument, like `--recon-model` or `--out-dir`, then it also
shows the available options (such as "emoca", "mediapipe", in case of `--recon-model`)
or the expected input type (like "PATH" in case of `--out-dir`).

If you, for example, would like to reconstruct your video, `my_vid.mp4`, using the
"mediapipe" model and store the output in the `recon/` directory, you'd run:

```console
$ medusa_videorecon my_vid.mp4 --recon-model mediapipe --out-dir recon/
```

Each CLI command follows its underlying Python function closely in terms of which
arguments it expects and which options it accepts, and all CLI defaults are the same
as the underlying Python function defaults.

## List of available CLI commands

Below, we outline and summarize all available Medusa CLI commands. For more info about
each command's options, run the command with the `--help` flag.

### `medusa_videorecon`

This command reconstructs the 3D face in each frame of the video. It assumes that the
video is in MP4 format and has the extension `.mp4`. It also assumes (for now) that
there is one, and only one, face present in each frame of the video.

This CLI command uses the Python function
[`medusa.recon.videorecon`](./python/recon/recon/index) under the hood.

### `medusa_videorender`

This command renders the time series of the reconstructed 3D face meshes as an MP4 video
or GIF (depending on the `--format` flag). It expects as input an HDF5 file with
reconstruction data and, if you want to render the reconstruction on top of the original
videon, a video file (e.g., `--video my_video.mp4`).

This CLI command uses the `render_video` method from the
[`medusa.core.Data4D`](./python/containers/fourD/index) class.

### `medusa_download_ext_data`

This command downloads external data necessary for some detection and reconstruction
models. As explained in the [installation instructions](../getting_started/installation),
you need to create an account on the [FLAME website](https://flame.is.tue.mpg.de/) first.
