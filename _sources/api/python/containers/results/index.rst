:py:mod:`medusa.containers.results`
===================================

.. py:module:: medusa.containers.results

.. autoapi-nested-parse::

   A very hack implementation of a container to store results from processing
   multiple batches of images.



Module Contents
---------------

.. py:class:: BatchResults(n_img=0, device=DEVICE, **kwargs)

   A container to store and process results from processing multiple
   batches of inputs/images.

   :param n_img: Number of images processed thus far
   :type n_img: int
   :param device: Device to store/process the data on (either 'cpu' or 'cuda')
   :type device: str
   :param \*\*kwargs: Other data that will be set as attributes

   .. py:method:: add(**kwargs)

      Add data to the container.

      :param \*\*kwargs: Any data that will be added to the


   .. py:method:: concat(n_max=None)

      Concatenate results form multiple batches.

      :param n_max: Whether to only return ``n_max`` observations per attribute
                    (ignored if ``None``)
      :type n_max: None, int


   .. py:method:: sort_faces(attr='lms', dist_threshold=250)

      'Sorts' faces using the ``medusa.tracking.sort_faces`` function (and
      performs some checks of the data).

      :param attr: Name of the attribute that needs to be used to sort the faces
                   (e.g., 'lms' or 'v')
      :type attr: str
      :param dist_threshold: Euclidean distance between two sets of landmarks/vertices that we consider
                             comes from two different faces (e.g., if ``d(lms1, lms2) >= dist_treshold``,
                             then we conclude that face 1 (``lms1``) is a different from face 2 (``lms2``)
      :type dist_threshold: int, float

      :returns: **face_idx** -- The face IDs associate with each detection
      :rtype: torch.tensor


   .. py:method:: filter_faces(present_threshold=0.1)


   .. py:method:: to_dict(exclude=None)


   .. py:method:: visualize(f_out, imgs, video=False, show_cropped=False, face_id=None, fps=24, crop_size=(224, 224), template=None, **kwargs)

      Visualizes the detection/cropping results aggregated by the
      BatchResults object.

      :param f_out: Path of output image/video
      :type f_out: str, Path
      :param imgs: A tensor with the original (uncropped images); can be a batch of images
                   or a single image
      :type imgs: torch.tensor
      :param video: Whether to output a video or image (grid)
      :type video: bool
      :param show_cropped: Whether to visualize the cropped image or the original image
      :type show_cropped: bool
      :param face_id: Should be None (used in recursive call)
      :type face_id: None
      :param fps: Frames per second of video (only relevant if ``video=True``)
      :type fps: int
      :param crop_size: Size of cropped images
      :type crop_size: tuple[int]
      :param template: Template used in aligment (optional)
      :type template: torch.tensor



