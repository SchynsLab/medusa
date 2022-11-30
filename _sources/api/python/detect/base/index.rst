:py:mod:`medusa.detect.base`
============================

.. py:module:: medusa.detect.base


Module Contents
---------------

.. py:class:: BaseDetectionModel


.. py:class:: DetectionResults(n_img, conf=None, bbox=None, lms=None, img_idx=None, device=DEVICE)

   .. py:method:: __len__()


   .. py:method:: from_batches(batches)
      :classmethod:


   .. py:method:: sort(dist_threshold=200, present_threshold=0.1)


   .. py:method:: visualize(imgs, f_out, video=False, **kwargs)

      Creates an image with the estimated bounding box (bbox) on top of it.

      :param image: A numpy array with the original (uncropped images); can also be
                    a torch Tensor; can be a batch of images or a single image
      :type image: array_like
      :param bbox: A numpy array with the bounding box(es) corresponding to the
                   image(s)
      :type bbox: np.ndarray
      :param f_out: If multiple images, a number (_xxx) is appended
      :type f_out: str, pathlib.Path



