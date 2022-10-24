from ._4D import Base4D, Flame4D, Mediapipe4D, Fan4D


MODEL2CLS = {
    "fan": Fan4D,
    "mediapipe": Mediapipe4D,
    "emoca-coarse": Flame4D,
    "emoca-dense": Flame4D,
    "deca-coarse": Flame4D,
    "deca-dense": Flame4D
}
