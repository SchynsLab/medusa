""" The ``core`` module contains two submodules, ``fourD`` and ``threeD`` that contain
classes to represent and manipulate 4D and 3D data, respectively. Data from different topologies
are represented by different classes, which all inherit from a common base class (e.g., ``Base4D``
for 4D data). For example, data reconstructed using the Mediapipe model is stored in an object of the
``Mediapipe4D`` class.
"""

from .fourD import Base4D, Flame4D, Mediapipe4D

MODEL2CLS = {
    "mediapipe": Mediapipe4D,
    "emoca-coarse": Flame4D,
    "emoca-dense": Flame4D,
    "deca-coarse": Flame4D,
    "deca-dense": Flame4D,
    "spectre-coarse": Flame4D,
    "spectre-dense": Flame4D
}
""" 
Allows to map string-based names of classes to the actual class.
"""

FLAME_MODELS = ['deca-coarse', 'deca-dense', 'emoca-coarse', 'emoca-dense',
                'spectre-coarse', 'spectre-dense']