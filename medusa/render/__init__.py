"""Top-level render package with a PyTorch-based renderer for (batches of) image(s)
and a utility class ``VideoRenderer`` for easy rendering of videos from ``Data4D`` objects,
as well as a class ``Overlay`` for creating vertex-based overlays."""

from .overlay import Overlay
from .video import VideoRenderer
from .image import PytorchRenderer
