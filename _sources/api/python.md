# Python interface

Medusa's Python interface allows you to use its functionality in Python. The package
contains several submodules, of which the most important ones are:

## [`medusa.core`](./python/core/index)

The `core` module contains the core data classes used in Medusa. The most important class
is the `BaseData` class. This class contains most of the functionality to initialize,
transform, visualize, load, and save reconstruction data. The `BaseData` class should 
not be used by itself; instead, use one of the other data classes that inherit from
the `BaseData` class:

* `FlameData`: for data from reconstruction models that use the [FLAME](https://flame.is.tue.mpg.de/)
topology (such as [EMOCA](./python/recon/emoca/index))

* `MediapipeData`: for data from the [Mediapipe](./python/recon/mpipe/index) reconstruction model

* `FANData`: for data from the [FAN-3D](./python/recon/fan/index) reconstruction model

## [`medusa.recon`](./python/recon/index)

The `recon` module contains implementations of several 3D face reconstruction models.
These implementations are basically wrappers around their original implementations,
which may have been simplified greatly (such as in the case of EMOCA). Each model is
implemented as a class that performs reconstruction when calling its `__call__` method.

For example, if you want to use the Mediapipe model to reconstruct the face in an image:

```python
from medusa.recon import Mediapipe
model = Mediapipe()

# Below is equivalent to: results = model.__call__(img)
results = model(img)
```

The following model classes are available:

* [`FAN`](./python/recon/fan/index)
* [`Mediapipe`](./python/recon/mpipe/index)
* [`EMOCA`](./python/recon/emoca/index)

## [`medusa.io`](./python/io/index)

The `io` module contains some code to make processing, reading, and writing videos
easier, implemented in the `VideoData` class. It also contains the `EpochsArray` class,
a subclass of the [MNE EpochsArray](https://mne.tools/stable/generated/mne.EpochsArray.html),
which allows you to read in epoched data from Medusa and save it as an MNE-compatible FIF
file.

## [`medusa.render`](./python/render/index)

The `render` module contains code to render (sequences of) 3D meshes as 2D images
(or videos). Under the hood, it makes heavy use of the awesome [pyrender](https://pyrender.readthedocs.io/)
package, which may not be the fastest renderer available, but its versatility and easy-of-use
makes it ideal for Medusa's purposes. 

The main object in the `io` module is the `Renderer` class, which makes it easy to
create scenes, add meshes, and subsequently render 3D face meshes.

## [`medusa.cli`](./cli)

The `cli` submodule contains the definition of the Medusa functions exposed to the
command-line interface. Its functionality is documented [here](./cli).

## [`medusa.preproc`](./python/preproc/index)

The bulk of Medusa's functionality in contained in the `preproc` module and its 
submodules. As its name suggests, it contains code to preprocess time series of 3D
face reconstructions.

The submodules are summarized below.

### [`medusa.preproc.recon`](./python/preproc/recon/index)

This submodule contains a single function, `videorecon`, that performs frame-by-frame 3D
reconstruction of a video and returns the results as an object of the appropriate
data class (e.g., `FlameData`).

### [`medusa.preproc.align`](./python/preproc/recon/index)

This submodule contains a single function, `align`, that performs alignment ("motion correction")
of a sequence of 3D meshes. It returns an instance of a data class of which the `v`
attribute (i.e., the vertices) have been aligned (either to a canonical model, like for
EMOCA and Mediapipe, or the mesh of the first time point, like for FAN). Note that this
function changes the `.space` attribute from `"world"` to `"local"`.

### [`medusa.preproc.resample`](./python/preproc/recon/index)

This submodule contains a single function, `resample`, that performs temporal resampling
of the 3D mesh time series by interpolating the data at a regular, evenly-spaced grid
with a specific sampling frequency. It returns an instance of the data class of which 
the data (`v` and, optionally, `mat`) has been resampled and the sampling frequency
(`.sf` attribute) has been changed accordingly.

### [`medusa.preproc.filter`](./python/preproc/recon/index)

This submodule contains a single function, `filter`, that performs temporal filtering
of the 3D mesh time series using a Butterworth filter. It returns an instance of the
appropriate data class.

### [`medusa.preproc.epoch`](./python/preproc/recon/index)

This submodule performs "epoching" on the time series of the reconstructed 3D face
meshes. Here, an "epoch" refers to an equal duration chunk of signal, usually time-locked
to repeated experimental events (such as stimulus onsets or button presses; definition
adapted from [MNE](https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html)).

It also (optionally) performs baseline correction and returns either a 4D numpy array
of shape $N$ (epochs) $\times\ T$ (number of time points of epoch) $\times\ V$ (number of vertices)
$\times\ 3$ (X, Y, Z), or an MNE-compatible [`EpochsArray`](https://mne.tools/stable/generated/mne.EpochsArray.html).

```{warning}
In Medusa, you can only use this functionality when you actually have an events-file with
stimulus/response/trial onsets (see [quickstart](../getting_started/quickstart) for more
information). 
```

## [`medusa.transform`](./python/transforms/index)

The `transform` module contains several functions to transform vertices between
different spaces (e.g., world space, NDC space, and raster space). This module is probably
not relevant for most users.

## [`medusa.utils`](./python/utils/index)

The `utils` module contains some utility functions, for example to create a nice
logger.