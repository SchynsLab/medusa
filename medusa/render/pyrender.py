"""Module with functionality to render 4D face mesh data.

The ``Renderer`` class is a high-level wrapper around functionality from the
excellent `pyrender <https://pyrender.readthedocs.io>`_ package [1]_.

.. [1] Matl, Matthew. *pyrender* [computer software]. https://github.com/mmatl/pyrender
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from OpenGL.GL import glLineWidth
from pyrender import (DirectionalLight, Mesh, Node, OffscreenRenderer,
                      OrthographicCamera, PerspectiveCamera, Scene)
from pyrender.constants import RenderFlags
from trimesh import Trimesh, visual

from ..constants import DEVICE
from .base import BaseRenderer


class PyRenderer(BaseRenderer):
    """A high-level wrapper around a pyrender-based renderer.

    Parameters
    ----------
    viewport : tuple[int]
        Desired output image size (width, height), in pixels; should match
        the original image (before cropping) that was reconstructed
    cam_type : str
        Either 'orthographic' (for Flame-based reconstructions) or
        'intrinsic' (for mediapipe reconstruction, i.e., a perspective camera)
    smooth : bool
        Whether to render a smooth mesh (by normal interpolation) or not
    wireframe : bool
        Whether to render a wireframe instead of a surface
    """

    def __init__(
        self,
        viewport,
        cam_mat=None,
        cam_type="orthographic",
        shading="flat",
        wireframe_opts=None,
        device=DEVICE,
    ):

        self.viewport = viewport
        self.cam_mat = cam_mat
        self.cam_type = cam_type
        self.shading = shading
        self.wireframe_opts = wireframe_opts
        self.device = device  # for compatibility only
        self._scene = self._create_scene()
        self._renderer = OffscreenRenderer(*self.viewport)  # actual pyrender renderer
        self._misc_config()

    def __str__(self):
        return "pyrender"

    def _create_scene(self):
        """Creates a simple scene with a camera and a directional light.

        The object (reconstructed face) is added when calling __call__.
        """
        w, h = self.viewport
        scene = Scene(bg_color=[0, 0, 0, 0], ambient_light=(255, 255, 255))

        if self.cam_type == "orthographic":
            camera = OrthographicCamera(xmag=1, ymag=1, znear=0.01)
        elif self.cam_type == "perspective":
            fov = 2 * np.arctan(self.viewport[1] / (2 * self.viewport[0]))  # in rad!
            camera = PerspectiveCamera(yfov=fov, znear=1.0, zfar=10000.0)
        else:
            raise ValueError(f"Unknown camera type {self.cam_type}")

        if self.cam_mat is None:
            cam_mat = np.eye(4)
        else:
            cam_mat = self.cam_mat
            if torch.is_tensor(cam_mat):
                cam_mat = cam_mat.cpu().numpy()

        camera_node = Node(camera=camera, matrix=cam_mat)
        scene.add_node(camera_node)
        light = DirectionalLight(intensity=5.0)
        scene.add_node(Node(light=light, matrix=cam_mat))

        return scene

    def _misc_config(self):

        if self.shading == "wireframe":

            if self.wireframe_opts is None:
                self.wireframe_opts = {}

            width = self.wireframe_opts.get("width", None)
            if width is not None:
                glLineWidth(width)

            color = self.wireframe_opts.get("color", None)
            if color is not None:
                self.wireframe_opts["color"] = np.array(color, dtype=np.float32)
            else:
                self.wireframe_opts["color"] = np.array(
                    [1.0, 0.0, 0.0, 1], dtype=np.float32
                )

    def __call__(self, v, tris, overlay=None, cmap_name="bwr"):
        """Performs the actual rendering.

        Parameters
        ----------
        v : np.ndarray
            A 2D array with vertices of shape V (nr of vertices) x 3
            (X, Y, Z)
        tris : np.ndarray
            A 2D array with 'triangles' of shape F (nr of faces) x 3
            (nr of vertices); should be integers
        overlay : np.ndarray
            A 1D array with overlay values (numpy floats between 0 and 1), one for each
            vertex or face
        cmap_name : str
            Name of (matplotlib) colormap; only relevant if ``overlay`` is not ``None``
        is_colors : bool
            If ``True``, then ``overlay`` is a V (of F) x 4 (RGBA) array; if ``False``,
            ``overlay`` is assumed to be a 1D array with floats betwee 0 and 1

        Returns
        -------
        img : np.ndarray
            A 3D array (with np.uint8 integers) of shape ``viewport[0]`` x
            ``viewport[1]`` x 3 (RGB)
        """

        v, tris = self._preprocess(v, tris, format="numpy")

        # if overlay is not None:
        #     overlay = overlay.squeeze()
        #     cmap = plt.get_cmap(cmap_name)

        #     if overlay.shape[0] == v.shape[0]:
        #         normals = mesh.vertex_normals
        #     elif overlay.shape[0] == f.shape[0]:
        #         normals = mesh.face_normals

        #     if overlay.ndim == 2:
        #         if overlay.shape[1] == 3:
        #             overlay = (overlay * normals).sum(axis=1)
        #         else:
        #             raise ValueError(f"Don't know what to do with overlay of shape {overlay.shape}!")

        #     from matplotlib.colors import TwoSlopeNorm
        #     vmin = overlay.min()
        #     vmax = overlay.max()
        #     if vmin != 0 and vmax != 0:
        #         tsn = TwoSlopeNorm(vmin=overlay.min(), vmax=overlay.max(), vcenter=0)
        #         overlay = tsn(overlay)
        #     overlay = cmap(overlay)
        #     if overlay.shape[0] == f.shape[0]:
        #         face_colors = overlay
        #         vertex_colors = None
        #     elif overlay.shape[0] == v.shape[0]:
        #         face_colors = None
        #         vertex_colors = overlay
        #     else:
        #         raise ValueError("Cannot infer whether overlay refers to vertices or "
        #                          "faces (polygons)!")

        #     mesh.visual = visual.create_visual(
        #         face_colors=face_colors,
        #         vertex_colors=vertex_colors,
        #         mesh=mesh
        #     )

        for i in range(v.shape[0]):

            mesh = Trimesh(v[i], tris)
            mesh = Mesh.from_trimesh(
                mesh,
                smooth=self.shading == "smooth",
                wireframe=self.shading == "wireframe",
            )

            if self.shading == "wireframe":
                mesh.primitives[0].material.baseColorFactor = self.wireframe_opts[
                    "color"
                ]

            mesh_node = Node(mesh=mesh)
            self._scene.add_node(mesh_node)

        with warnings.catch_warnings():
            # Silence OpenGL DeprecationWarning
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            img, _ = self._renderer.render(self._scene, flags=RenderFlags.RGBA)

        self._scene.mesh_nodes.clear()
        img = torch.as_tensor(img.copy(), device=self.device)
        return img

    def close(self):
        """Closes the OffScreenRenderer object."""
        self._renderer.delete()
        self._scene.clear()
