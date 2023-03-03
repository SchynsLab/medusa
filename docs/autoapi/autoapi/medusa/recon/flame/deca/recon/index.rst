:py:mod:`medusa.recon.flame.deca.recon`
=======================================

.. py:module:: medusa.recon.flame.deca.recon

.. autoapi-nested-parse::

   Module with different FLAME-based 3D reconstruction models, including DECA.

   [1]_, EMOCA [2]_, and spectre [3]_.

   All model classes inherit from a common base class, ``FlameReconModel`` (see
   ``flame.base`` module).

   .. [1] Feng, Y., Feng, H., Black, M. J., & Bolkart, T. (2021). Learning an animatable detailed
          3D face model from in-the-wild images. ACM Transactions on Graphics (ToG), 40(4), 1-13.
   .. [2] Danecek, R., Black, M. J., & Bolkart, T. (2022). EMOCA: Emotion Driven Monocular
          Face Capture and Animation. arXiv preprint arXiv:2204.11312.
   .. [3] Filntisis, P. P., Retsinas, G., Paraperas-Papantoniou, F., Katsamanis, A., Roussos, A., & Maragos, P. (2022).
          Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos.
          arXiv preprint arXiv:2207.11094.

   For the associated license, see license.md.



Module Contents
---------------

.. py:class:: DecaReconModel(name, orig_img_size=None, device=DEVICE)



   A 3D face reconstruction model that uses the FLAME topology.

   At the moment, six different models are supported: 'deca-coarse', 'deca-dense',
   'emoca-coarse', 'emoca-dense', 'spectre-coarse', and 'spectre-dense'

   :param name: Either 'deca-coarse', 'deca-dense', 'emoca-coarse', or 'emoca-dense'
   :type name: str
   :param orig_img_size: Original (before cropping!) image dimensions of video frame (width, height);
                         needed for baking in translation due to cropping; if not set, it is assumed
                         that the image is not cropped!
   :type orig_img_size: tuple
   :param device: Either 'cuda' (uses GPU) or 'cpu'
   :type device: str
