# Command-line interface

If you're familiar with using terminals and command line interfaces (CLI), Medusa's
CLI might be useful for you. The CLI exposes Medusa's most important functionality
to the command line, which can be used after installing the Python package. Alternatively,
you can use the [Python API](./python).

All CLI commands start with `medusa_` followed by the operation that it exposes.
For example, `medusa_videorecon` is a command-line interface of the package's Python
function [`videorecon`](./python/preproc/recon/index). The arguments and options of each CLI
command can be inspected by running the command with a single option `--help`. For
example:

```console
$ medusa_videorecon --help

Usage: medusa_videorecon [OPTIONS] VIDEO_PATH

Options:
  --events-path FILE
  -r, --recon-model-name [emoca|mediapipe|FAN-3D]
  -c, --cfg TEXT                  Path to recon config file
  --device [cpu|cuda]             Device to run recon on
  -o, --out-dir PATH              Output directory
  --render-recon                  Plot recon on video background
  --render-on-video               Plot recon on video background
  --render-crop                   Render cropping results
  -n, --n-frames INTEGER          Number of frames to reconstruct
  --help                          Show this message and exit.
```

For example, the `medusa_videorecon` command has a single mandatory argument,
`VIDEO_PATH`, and several (non-mandatory) options, like `--events-path` and 
`--recon-model-name`. If the option accepts an argument, like `--recon-model-name` or
`--out-dir`, then it also shows the available options (such as "emoca", "mediapipe", or
"FAN-3D", in case of `--recon-model-name`) or the expected input type (like "PATH" in 
case of `--out-dir`). 

If you, for example, would like to reconstruct your video, 
`my_vid.mp4`, using the "mediapipe" model and store the output in the `recon/` directory,
you'd run:

```console
$ medusa_videorecon my_vid.mp4 --recon-model-name mediapipe --out-dir recon/
```

In addition, there may be some options which are not followed by an argument, like
`--render-recon` (which are not followed by available options or expected input type).
So, if you'd want to run the same reconstruction as the previous command, but this time
also render the reconstruction (`--render-recon`) on top of the input video 
(`--render-on-video`), you'd run:

```console
$ medusa_videorecon my_vid.mp4 --recon-model-name mediapipe --out-dir recon/ --render-recon --render-on-video
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
[`medusa.preproc.recon.videorecon`](./python/preproc/recon/index) under the hood.

### `medusa_align`

This command spatially aligns the reconstructed 3D meshes from different time points,
which is also known as "motion correction" in the functional MRI literature. It expects
as input an HDF5 file (extension: `.h5`) with reconstruction data, as created by running
the `medusa_videorecon` command.

If a "local-to-world" matrix is known for each time point (as provided by the EMOCA and
Mediapipe reconstruction models), each mesh is aligned to the underlying canonical model
by applying the inverse matrix (i.e., the "world-to-local" matrix) to the vertices.

If these matrices are not known, each mesh ($V_{i}$) is aligned to the mesh of the first
time point ($V_{1}$) using the ICP algorithm {cite:p}`arun1987least` (from
[trimesh](https://trimsh.org/trimesh.registration.html)) or Umeyama algorithm 
{cite:p}`umeyama1991least` (from 
[scikit-image](https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py#L91)).

This CLI command uses the Python function
[`medusa.preproc.align.align`](./python/preproc/align/index) under the hood.

```{note}
If you want to separate global face movements (translation and rotation) from local face 
movements (facial soft tissue movement due to muscle activations), you need to run this
algorithm.
```

### `medusa_resample`

This command temporally resamples the time series of reconstructed 3D face meshes. It
expects as input an HDF5 file with reconstruction data. The command can be used to resamle
the data to a regular period and/or to upsample the data (by setting the
`--sampling-freq` higher the the video's sampling frequency, or FPS).

This CLI command uses the Python function
[`medusa.preproc.resample.resample`](./python/preproc/resample/index) under the hood.

```{note}
For videos with a regular sampling frequency (i.e., the same time period between
frames), running this command is not strictly necessary.
```

### `medusa_filter`

This command performs temporal filtering on the time series of the reconstructed 3D face
meshes. It expects as input an HDF5 file with reconstruction data. It uses a
[Butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter) to perform low-
and/or high-pass filtering. The cutoffs are expected to be given in hertz.

It uses a [Butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter)
to perform low- and/or high-pass filtering. The cutoffs are expected to be given in
hertz.

This CLI command uses the Python function
[`medusa.preproc.filter.filter`](./python/preproc/filter/index) under the hood.

### `medusa_epoch`

This command performs "epoching" on the time series of the reconstructed 3D face
meshes. It expects as input an HDF5 file with reconstruction data. Here, an "epoch" refers to
an equal duration chunk of signal, usually time-locked to repeated experimental events
(such as stimulus onsets or button presses; definition adapted from
[MNE](https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html)).

```{warning}
In Medusa, you can only use this functionality when you actually have an events-file with
stimulus/response/trial onsets (see [quickstart](../getting_started/quickstart) for more
information). 
```

The result of the epoching operation is a 4D numpy array of shape $N$ (epochs) $\times\ T$
(number of time points of epoch) $\times\ V$ (number of vertices) $\times\ 3$ (X, Y, Z).
The CLI command actually converts this data into an MNE-compatible structure
(an [`EpochsArray`](https://mne.tools/stable/generated/mne.EpochsArray.html)) and saves
it as a FIF file (extension: `.fif`). This file can then be loaded using MNE 
(with [`mne.read_epochs`](https://mne.tools/stable/generated/mne.read_epochs.html#mne.read_epochs))
to be further analyzed.

This CLI command uses the Python function
[`medusa.preproc.epoch.epoch`](./python/preproc/epoch/index) under the hood.

### `medusa_videorender`

This command renders the time series of the reconstructed 3D face meshes as an MP4 video
or GIF (depending on the `--format` flag). It expects as input an HDF5 file with
reconstruction data and, if you want to render the reconstruction on top of the original
videon, a video file (e.g., `--video my_video.mp4`).

This CLI command uses the `render_video` method from the
[`medusa.core.BaseData`](./python/core/index) class, which in turn uses the 
[`medusa.render.Renderer`](./python/render/index) class (a wrapper around a
[pyrender](https://pyrender.readthedocs.io/) renderer).
