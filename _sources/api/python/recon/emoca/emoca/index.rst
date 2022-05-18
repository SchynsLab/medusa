:py:mod:`medusa.recon.emoca.emoca`
==================================

.. py:module:: medusa.recon.emoca.emoca


Module Contents
---------------

.. py:class:: EMOCA(img_size, cfg=None, device='cuda')

   Bases: :py:obj:`torch.nn.Module`

   An wrapper around the EMOCA face reconstruction model.

   :param img_size: Original (before cropping!) image dimensions of
                    video frame (width, height); needed for baking in
                    translation due to cropping
   :type img_size: tuple
   :param cfg: Path to YAML config file. If `None` (default), it
               will use the package's default config file.
   :type cfg: str
   :param device: Either 'cuda' (uses GPU) or 'cpu'
   :type device: str

   .. attribute:: tform

      A 3x3 numpy array with the cropping transformation matrix;
      needs to be set before running the actual reconstruction!

      :type: np.ndarray

   .. rubric:: Examples

   To initialize an EMOCA model:

   >>> from medusa.data import get_example_frame
   >>> img_size = get_example_frame().shape[:2]
   >>> model = EMOCA(img_size, device='cpu')  # use 'cuda' whenever possible!

   .. py:attribute:: benchmark
      :annotation: = True

      

   .. py:method:: __call__(self, image)

      Performs reconstruction of the face as a list of landmarks (vertices).

      :param image: A 4D (1 x 3 x w x h) ``torch.Tensor`` representing a RGB image (and a
                    batch dimension of 1)
      :type image: torch.Tensor

      :returns: **out** -- A dictionary with two keys: ``"v"``, the reconstructed vertices (5023 in
                total) and ``"mat"``, a 4x4 Numpy array representing the local-to-world
                matrix
      :rtype: dict

      .. rubric:: Notes

      Before calling ``__call__``, you *must* set the ``tform`` attribute to the
      estimated cropping matrix (see example below). This is necessary to encode the
      relative position and scale of the bounding box into the reconstructed vertices.

      .. rubric:: Examples

      To reconstruct an example, call the ``EMOCA`` object, but make sure to set the
      ``tform`` attribute first:

      >>> from medusa.data import get_example_frame
      >>> from medusa.recon import FAN
      >>> img = get_example_frame()
      >>> model = EMOCA(img.shape[:2], device='cpu')  # doctest: +SKIP
      >>> fan = FAN(lm_type='2D')   # need FAN for cropping!
      >>> cropped_img = fan.prepare_for_emoca(img)
      >>> model.tform = fan.tform.params  # crucial!
      >>> out = model(cropped_img)  # doctest: +SKIP
      >>> out['v'].shape    # doctest: +SKIP
      (5023, 3)
      >>> out['mat'].shape  # doctest: +SKIP
      (4, 4)



