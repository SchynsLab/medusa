"""Top-level rendering module, containg Medusa's two main renderers:

* ``PyRenderer`` (based on the ``pyrender`` package)
* ``PytorchRenderer (based on the ``pytorch3d`` package, if available)
"""

from .pyrender import PyRenderer
from .pytorch3d import PytorchRenderer
# try:
#     from .pytorch3d import PytorchRenderer
# except ImportError as e:
#     # pytorch3d not available
#     pass
