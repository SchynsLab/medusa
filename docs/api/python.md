# Python interface

Medusa's Python interface allows you to use its functionality in Python. The API
reference was automatically build from the package's docstrings using [Sphinx AutoAPI](https://sphinx-autoapi.readthedocs.io).

The package contains several subpackages, of which the most important ones are:

## [`medusa.containers`](./python/containers/index)

The `containers` module contains the core data classes used in Medusa. The most important
class is the `Data4D` class. This class contains most of the functionality to store,
transform, load, and save 4D reconstruction data.

## [`medusa.recon`](./python/recon/index)

The `recon` module contains implementations of several 3D face reconstruction models.
These implementations are basically wrappers around their original implementations,
which may have been simplified greatly (such as in the case of EMOCA) as they only need
to perform inference (not training). Each model is implemented as a class that performs
reconstruction or a batch of images by calling its `__call__` method.

For example, if you want to use the Mediapipe model to reconstruct the face in an image:

```python
from medusa.recon import Mediapipe
model = Mediapipe()

# Below is equivalent to: results = model.__call__(img)
results = model(img)
```

## [`medusa.io`](./python/io/index)

The `io` module contains some code to make processing, reading, and writing videos
easier. Its most important classes are `VideoLoader` and `VideoWriter`.

## [`medusa.render`](./python/render/index)

The `render` module contains code to render (sequences of) 3D meshes as 2D images
(or videos). As the name suggests, the `PytorchRenderer` is a wrapper around functionality
from the [pytorch3d](https://pytorch3d.org/) package {cite}`ravi2020pytorch3d`. The
`VideoRenderer` is a high-level class to easily render 4D reconstruction data as a video.

## [`medusa.detect`](./python/detect/index)

The `detect` module contains two classes for face detection: `YunetDetector` {cite}`facedetect-yu` and
`SCRFDetector` {cite}`guo2021sample`. The `YunetDetector` class assumes that the Python
package `python-opencv` is installed (which isn't installed by default when installing
Medusa).

## [`medusa.crop`](./python/crop/index)

The `crop` module contains two high-level classes that perform (batched) image cropping:
`BboxCropModel` (crop images based on an estimated bounding box) and `AlignCropModel`
(crops images based on landmark alignment to a template).

## [`medusa.analysis`](./python/analysis/index)

The `analysis` module contains some convenience functions for common analyses. (Work in progress.)

## [`medusa.epoch`](./python/epoch/index)

The `epoch` module contains functionality to create epochs from `Data4D` files.
