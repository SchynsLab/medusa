""" This module contains several high-level functions for preprocessing time series
of 3D meshes. All functions take as a (mandatory) input argument a ``*Data`` object
(like ``FlameData`` or ``MediapipeData``), with the exception of the ``videorecon``
function, which needs a MP4 video as input. All functions output a (processed)
``*Data`` object again.

The recommended order in which to run the preprocessing functions are:

1. ``videorecon``
2. ``align``
3. ``resample``
4. ``filter``
5. ``epoch``
"""

from .recon import videorecon
from .align import align
from .epoch import epoch
from .filter import filter
from .resample import resample