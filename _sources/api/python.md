# Python interface

Medusa's Python interface allows you to use its functionality in Python. The package
contains several submodules, of which the most important ones are:

## [`medusa.containers`](./python/containers/index)

The `containers` module contains the core data classes used in Medusa. The most important class
is the `Data4D` class. This class contains most of the functionality to initialize,
transform, visualize, load, and save 4D reconstruction data.

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

* [`Mediapipe`](./python/recon/mpipe/index)
* [`DECA/EMOCA/Spectre`](./python/recon/flame/deca/index)
* ['MIECA`](./python/recon/flame/mica/index)

## [`medusa.io`](./python/io/index)

The `io` module contains some code to make processing, reading, and writing videos
easier, implemented in the `VideoLoader` class.

## [`medusa.render`](./python/render/index)

The `render` module contains code to render (sequences of) 3D meshes as 2D images
(or videos). Under the hood, it makes heavy use of the awesome [pyrender](https://pyrender.readthedocs.io/)
package, which may not be the fastest renderer available, but its versatility and easy-of-use
makes it ideal for Medusa's purposes. Alternatively, on Linux, Medusa can also use
a Pytorch3D-based renderer.

## [`medusa.cli`](./cli)

The `cli` submodule contains the definition of the Medusa functions exposed to the
command-line interface. Its functionality is documented [here](./cli).

## [`medusa.transforms`](./python/transforms/index)

The `transform` module contains several functions to transform vertices between
different spaces (e.g., world space, NDC space, and raster space). This module is probably
not relevant for most users.
