:py:mod:`medusa.render.video`
=============================

.. py:module:: medusa.render.video

.. autoapi-nested-parse::

   A module with functionality to easily render a 4D mesh to a video using the
   ``VideoRender`` class.



Module Contents
---------------

.. py:class:: VideoRenderer(shading='flat', lights=None, background=(0, 0, 0), loglevel='INFO')

   Renders a 4D mesh to a video.

   :param shading: Type of shading ('flat', 'smooth', or 'wireframe'; latter only when using
                   'pyrender')
   :type shading: str
   :param lights: Lights to use in rendering; if None, a default PointLight will be used
   :type lights: None, str, or pytorch3d Lights class
   :param background: Either a path to a video to be used as a background (or an initialized
                      ``VideoLoader``) or a 3-tuple of RGB values (between 0 and 255) to use as uniform
                      color background; default is a uniform black background
   :type background: str, Path, VideoLoader, tuple
   :param loglevel: Logging level (e.g., "INFO", "DEBUG", etc.)
   :type loglevel: str

   .. py:method:: render(f_out, data4d, overlay=None, **kwargs)

      Renders the sequence of 3D meshes from a Data4D object as a video.

      :param f_out: Filename of output
      :type f_out: str
      :param data4d: A ``Data4D`` object
      :type data4d: Data4D
      :param overlay: Optional overlay to render on top of rendered mesh
      :type overlay: torch.tensor, TextureVertex
      :param video: Path to video, in order to render face on top of original video frames
      :type video: str
      :param \*\*kwargs: Keyword arguments, which may include "viewport", "device", "cam_mat", "fps",
                         and "n_frames". If not specified, these will be inferred from the supplied
                         Data4D object.



