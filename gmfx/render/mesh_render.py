# From: https://github.com/1996scarlet/Dense-Head-Pose-Estimation

import ctypes
import numpy as np


class TrianglesMeshRender:

    def __init__(self, clibs, tri, light=[1, 1, 5],
                 direction=[0.6, 0.6, 0.6], ambient=[0.6, 0.5, 0.4]):

        self._clibs = ctypes.CDLL(clibs)

        self._tri = np.load(tri)
        self._tri_nums = self._tri.shape[0]
        self._tri = np.ctypeslib.as_ctypes(self._tri)

        self._light = np.array(light, dtype=np.float32)
        self._light = np.ctypeslib.as_ctypes(self._light)

        self._direction = np.array(direction, dtype=np.float32)
        self._direction = np.ctypeslib.as_ctypes(self._direction)

        self._ambient = np.array(ambient, dtype=np.float32)
        self._ambient = np.ctypeslib.as_ctypes(self._ambient)

    def render(self, verts, bg):
        self._clibs._render(
            self._tri, self._tri_nums,
            self._light, self._direction, self._ambient,
            np.ctypeslib.as_ctypes(verts),
            verts.shape[0],
            np.ctypeslib.as_ctypes(bg),
            bg.shape[0], bg.shape[1]
        )