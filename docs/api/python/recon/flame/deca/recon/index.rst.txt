:py:mod:`medusa.recon.flame.deca.recon`
=======================================

.. py:module:: medusa.recon.flame.deca.recon

.. autoapi-nested-parse::

   Module with different FLAME-based 3D reconstruction models, including
   DECA [1]_, EMOCA [2]_, and spectre [3_].

   All model classes inherit from a common base class, ``FlameReconModel`` (see ``flame.base`` module).

   .. [1] Feng, Y., Feng, H., Black, M. J., & Bolkart, T. (2021). Learning an animatable detailed
          3D face model from in-the-wild images. *ACM Transactions on Graphics (ToG), 40*(4), 1-13.
   .. [2] Danecek, R., Black, M. J., & Bolkart, T. (2022). EMOCA: Emotion Driven Monocular
          Face Capture and Animation. *arXiv preprint arXiv:2204.11312*.
   .. [3] Filntisis, P. P., Retsinas, G., Paraperas-Papantoniou, F., Katsamanis, A., Roussos, A., & Maragos, P. (2022).
          Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos.
          *arXiv preprint arXiv:2207.11094*.



Module Contents
---------------

.. py:data:: logger
   

   

.. py:class:: DecaReconModel(name, img_size=None, device='cuda', crop_mat=None)

   Bases: :py:obj:`medusa.recon.flame.base.FlameReconModel`

   A 3D face reconstruction model that uses the FLAME topology.

   At the moment, four different models are supported: 'deca-coarse', 'deca-dense',
   'emoca-coarse', and 'emoca-dense'.

   :param name: Either 'deca-coarse', 'deca-dense', 'emoca-coarse', or 'emoca-dense'
   :type name: str
   :param img_size: Original (before cropping!) image dimensions of video frame (width, height);
                    needed for baking in translation due to cropping; if not set, it is assumed
                    that the image is not cropped!
   :type img_size: tuple
   :param device: Either 'cuda' (uses GPU) or 'cpu'
   :type device: str
   :param crop_mat: A b (batch_size) x 3 x 3 tensor which contains the cropping matrices
                    for the images; needed to transform the vertices to conform to the original
                    image space
   :type crop_mat: torch.Tensor

   .. attribute:: crop_mat

      A 3x3 numpy array with the cropping transformation matrix;
      needs to be set before running the actual reconstruction!

      :type: np.ndarray

   .. py:attribute:: benchmark
      :annotation: = True

      

   .. py:method:: __call__(images)

      Performs reconstruction of the face as a list of landmarks (vertices).

      :param images: A 4D (batch_size x 3 x 224 x 224) ``torch.Tensor`` representing batch of
                     RGB images cropped to 224 (h) x 224 (w)
      :type images: torch.Tensor

      :returns: **dec_dict** -- A dictionary with two keys: ``"v"``, a numpy array with reconstructed vertices
                (5023 for  'coarse' models or 59315 for 'dense' models) and ``"mat"``, a
                4x4 numpy array representing the local-to-world matrix
      :rtype: dict

      .. rubric:: Notes

      Before calling ``__call__``, you *must* set the ``crop_mat`` attribute to the
      estimated cropping matrix if you want to be able to render the reconstruction
      in the original image (see example below)

      .. rubric:: Examples

      To reconstruct an example, call the ``EMOCA`` object, but make sure to set the
      ``crop_mat`` attribute first:

      >>> from medusa.data import get_example_frame
      >>> from medusa.crop import FanCropModel
      >>> img = get_example_frame()
      >>> crop_model = FanCropModel(device='cpu')
      >>> cropped_img, crop_mat = crop_model(img)
      >>> recon_model = DecaReconModel(name='emoca-coarse', device='cpu')
      >>> recon_model.crop_mat = crop_mat
      >>> out = recon_model(cropped_img)
      >>> out['v'].shape
      (1, 5023, 3)
      >>> out['mat'].shape
      (1, 4, 4)



