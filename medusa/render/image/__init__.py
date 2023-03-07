"""Top-level mesh-to-image rendering module, containg Medusa's (currently) only
renderer:

* ``PytorchRenderer (based on the ``pytorch3d`` package, if available)
"""

try:
    from .pytorch3d import PytorchRenderer
except ImportError as e:
    # pytorch3d not available
    pass
