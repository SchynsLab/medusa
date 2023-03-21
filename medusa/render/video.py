"""A module with functionality to easily render a 4D mesh to a video using the
``VideoRender`` class."""

import torch
from pathlib import Path

from ..io import VideoLoader, VideoWriter
from ..log import tqdm_log
from ..defaults import LOGGER


class VideoRenderer:
    """Renders a 4D mesh to a video.

    Parameters
    ----------
    render_cls : PytorchRenderer
        Renderer class to use (at the moment, only PytorchRenderer is supported)
    shading : str
        Type of shading ('flat', 'smooth', or 'wireframe'; latter only when using
        'pyrender')
    lights : None, str, or pytorch3d Lights class
        Lights to use in rendering; if None, a default PointLight will be used
    """

    def __init__(self, shading="flat", lights=None, loglevel="INFO"):
        """Initializes a VideoRenderer object."""
        self.shading = shading
        self.lights = lights
        self._renderer = None  # lazy init

        LOGGER.setLevel(loglevel)

    def __call__(self, f_out, data, overlay=None, video=None, **kwargs):
        """Renders the sequence of 3D meshes from a Data4D object as a video.

        Parameters
        ----------
        f_out: str
            Filename of output
        data : Data4D
            A ``Data4D`` object
        overlay : torch.tensor, TextureVertex
            Optional overlay to render on top of rendered mesh
        video : str
            Path to video, in order to render face on top of original video frames
        **kwargs
            Keyword arguments, which may include "viewport", "device", "cam_mat", "fps",
            and "n_frames". If not specified, these will be inferred from the supplied
            Data4D object.
        """

        try:
            from . import PytorchRenderer
        except ImportError:
            raise ImportError("pytorch3d not installed; cannot render!")

        if data._infer_topo() == 'mediapipe':
            cam_type = "perspective"
        else:
            cam_type = "orthographic"

        # If parameters are defined in kwargs, use those; otherwise get from data4d
        device = kwargs.get("device", data.device)
        cam_mat = kwargs.get("cam_mat", data.cam_mat)
        viewport = kwargs.get("viewport", data.video_metadata.get('img_size'))

        if self._renderer is None:
            # Initialize renderer
            self._renderer = PytorchRenderer(
                viewport, cam_mat, cam_type, self.shading, self.lights, device
            )
        elif ~torch.allclose(cam_mat, self._renderer.cam_mat):
            # cam_mat has changed; we need to re-initialize the renderer!
            self._renderer = PytorchRenderer(
                viewport, cam_mat, cam_type, self.shading, self.lights, device
            )

        # Define a VideoLoader if we want to render mesh on top of video
        if isinstance(video, VideoLoader):
            reader = video
            if reader.batch_size != 1:
                raise ValueError("Batch size of VideoLoader must be 1 for rendering!")
        elif isinstance(video, (str, Path)):
            # assume path to video
            reader = VideoLoader(video, batch_size=1, device=device)
        else:
            reader = None

        fps = kwargs.get("fps", data.video_metadata["fps"])
        n_frames = kwargs.get("n_frames", data.video_metadata["n_img"])
        writer = VideoWriter(str(f_out), fps=fps)  # writes out images to video stream

        # Loop of number of frames from original video (or n_frames if specified)
        iter_ = tqdm_log(range(n_frames), LOGGER, desc="Render mesh")
        for i_img in iter_:

            if video is not None:
                # Get background from video
                background = next(iter(reader))
            else:
                # Blank background is white by default; make parameter?
                w, h = viewport
                background = torch.full((h, w, 3), 255, dtype=torch.uint8, device=device)

            # Which meshes belong to this frame?
            img_idx = data.img_idx == i_img

            if img_idx.sum() == 0:
                # No faces in this frame! Just render background
                img = background
            else:
                # Check whether there is an overlay (texture)
                this_overlay = overlay
                if overlay is not None:
                    # Check if it's a single overlay (which should be used for every
                    # frame) or a batch of overlays (one per frame)
                    if torch.is_tensor(overlay):
                        if overlay.shape[0] == data.v.shape[0]:
                            this_overlay = overlay[img_idx]
                        elif overlay.shape[0] == data.v.shape[1]:
                            this_overlay = overlay

                # Render mesh and alpha blend with background
                this_v = data.v[img_idx]
                img = self._renderer(this_v, data.tris, overlay=this_overlay)
                img = self._renderer.alpha_blend(img, background, face_alpha=1)

            # Add rendered image to video writer
            writer.write(img)

        writer.close()

        if video is not None:
            reader.close()

    def close(self):
        """Closes the renderer."""
        if self._renderer is not None:
            self._renderer.close()
