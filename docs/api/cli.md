# Command-line interface

If you're familiar with using terminals and command line interfaces (CLI), Medusa's
CLI might be useful for you. The CLI exposes Medusa's most important functionality
to the command line, which can be used after installing the Python package. All CLI
commands start with `medusa_` followed by the operation that it exposes. For example,
`medusa_videorecon` is a command-line interface of the package's Python function
[`videorecon`](./python/preproc/recon). The arguments and options of each CLI command
can be inspected by running the command with a single option `--help`, like:

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
case of `--out-dir`). If you, for example, would like to reconstruct your video, 
`my_vid.mp4`, using the "mediapipe" model and store the output in the `recon/` directory,
you'd run:

```console
medusa_videorecon my_vid.mp4 --recon-model-name mediapipe --out-dir recon/
```

In addition, there may be some options which are not followed by an argument, like
`--render-recon` (which are not followed by available options or expected input type).
So, if you'd want to run the same reconstruction as the previous command, but this time
also render the reconstruction (`--render-recon`) on top of the input video 
(`--render-on-video`), you'd run:

```console
medusa_videorecon my_vid.mp4 --recon-model-name mediapipe --out-dir recon/ --render-recon --render-on-video
```

Each CLI command follows its underlying Python function closely in terms of which
arguments it expects and which options it accepts.