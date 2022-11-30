:py:mod:`medusa.recon.flame.data.example_data`
==============================================

.. py:module:: medusa.recon.flame.data.example_data


Module Contents
---------------

.. py:function:: get_example_img(load=False)

   Loads a test image from the ``flame/data`` directory (an image generated on
   https://this-person-does-not-exist.com).

   :param load: Whether to load the image into memory as a numpy array or just return the
                filepath
   :type load: bool
   :param Returns:
   :param img: Image or path to image, depending on the ``load`` parameter
   :type img: pathlib.Path, np.ndarray


