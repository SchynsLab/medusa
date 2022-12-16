"""Module with functionality to render 4D face mesh data.

The ``Renderer`` class is a high-level wrapper around functionality from the
excellent `pyrender <https://pyrender.readthedocs.io>`_ package [1]_.

.. [1] Matl, Matthew. *pyrender* [computer software]. https://github.com/mmatl/pyrender
"""

import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from OpenGL.GL import glLineWidth
from pyrender import (DirectionalLight, IntrinsicsCamera, Mesh, Node,
                      OffscreenRenderer, OrthographicCamera, Scene)
from pyrender.constants import RenderFlags
from trimesh import Trimesh, visual


class Renderer:
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

    def __init__(self, viewport, cam_mat=None, cam_type="orthographic", shading='flat',
                 wireframe_opts=None):

        self.viewport = viewport
        self.cam_mat = cam_mat
        self.cam_type = cam_type
        self.shading = shading
        self.wireframe_opts = wireframe_opts
        self._scene = self._create_scene()
        self._renderer = OffscreenRenderer(*self.viewport)  # actual pyrender renderer
        self._misc_config()

    def _create_scene(self):
        """Creates a simple scene with a camera and a directional light.

        The object (reconstructed face) is added when calling __call__.
        """
        w, h = self.viewport
        scene = Scene(bg_color=[0, 0, 0, 0], ambient_light=(255, 255, 255))

        if self.cam_type == "orthographic":
            camera = OrthographicCamera(xmag=1, ymag=1)
        elif self.cam_type == "intrinsic":
            # Note to self: zfar might have to be increased if the face is
            # very small; will increase rendering time though
            fx, fy = w, w
            camera = IntrinsicsCamera(fx=fx, fy=fy, cx=w / 2, cy=h / 2, zfar=300)

        if self.cam_mat is None:
            cam_mat = np.eye(4)
        else:
            cam_mat = self.cam_mat

        camera_node = Node(camera=camera, matrix=cam_mat)
        scene.add_node(camera_node)
        light = DirectionalLight(intensity=7.5)
        scene.add_node(Node(light=light, matrix=cam_mat))

        return scene

    def _misc_config(self):

        if self.shading == 'wireframe':

            if self.wireframe_opts is None:
                self.wireframe_opts = {}

            width = self.wireframe_opts.get('width', None)
            if width is not None:
                glLineWidth(width)

            color = self.wireframe_opts.get('color', None)
            if color is not None:
                self.wireframe_opts['color'] = np.array(color, dtype=np.float32)
            else:
                self.wireframe_opts['color'] = np.array([1.0, 0.0, 0.0, 1], dtype=np.float32)

    def __call__(self, v, tris, overlay=None, cmap_name='bwr'):
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

        if torch.is_tensor(v):
            v = v.cpu().numpy()

        if torch.is_tensor(tris):
            tris = tris.cpu().numpy()

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

        if v.ndim == 2:
            v = v[None, ...]

        for i in range(v.shape[0]):
            mesh = Trimesh(v[i], tris)
            mesh = Mesh.from_trimesh(mesh, smooth=self.shading == 'smooth',
                                     wireframe=self.shading == 'wireframe')

            if self.shading == 'wireframe':
                mesh.primitives[0].material.baseColorFactor = self.wireframe_opts['color']

            mesh_node = Node(mesh=mesh)
            self._scene.add_node(mesh_node)

        with warnings.catch_warnings():
            # Silence OpenGL DeprecationWarning
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            img, _ = self._renderer.render(self._scene, flags=RenderFlags.RGBA)

        self._scene.mesh_nodes.clear()
        return img.copy()

    def alpha_blend(self, img, background, face_alpha=None):
        """Simple alpha blend of a rendered image and a background. The image
        (`img`) is assumed to be an RGBA image and the background
        (`background`) is assumed to be a RGB image. The alpha channel of the
        image is used to blend them together. The optional `threshold`
        parameter can be used to impose a sharp cutoff.

        Parameters
        ----------
        img : np.ndarray
            A 3D numpy array of shape height x width x 4 (RGBA)
        background : np.ndarray
            A 3D numpy array of shape height x width x 3 (RGB)
        """
        alpha = img[:, :, 3, None] / 255.0

        if face_alpha is not None:
            alpha[alpha > face_alpha] = face_alpha

        img = img[:, :, :3] * alpha + (1 - alpha) * background

        # TODO: add global alpha level for face
        return img.astype(np.uint8)

    def close(self):
        """Closes the OffScreenRenderer object."""
        self._renderer.delete()
        self._scene.clear()
