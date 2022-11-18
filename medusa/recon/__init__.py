""" Module with functionality to crop images based on landmarks estimated by the
``face_alignment`` package by `Adrian Bulat <https://www.adrianbulat.com/face-alignment>`_ [1]_.
or the ``insightface`` package [2]_. Note that the ``insightface`` package is not installed
by default. To install this package, 

.. [1] Bulat, A., & Tzimiropoulos, G. (2017). How far are we from solving the 2d & 3d
       face alignment problem?(and a dataset of 230,000 3d facial landmarks).
       In *Proceedings of the IEEE International Conference on Computer Vision*
       (pp. 1021-1030).

.. [2] Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). Retinaface:
       Single-shot multi-level face localisation in the wild. In Proceedings of the
       IEEE/CVF conference on computer vision and pattern recognition (pp. 5203-5212).
"""

from .base import BaseReconModel  
from .mpipe.mpipe import Mediapipe
from .flame import DecaReconModel, MicaReconModel
from .recon import videorecon
