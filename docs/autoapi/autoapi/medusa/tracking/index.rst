:py:mod:`medusa.tracking`
=========================

.. py:module:: medusa.tracking

.. autoapi-nested-parse::

   Module with functionality to 'track' faces, i.e., to associate the same face
   across detections/reconstructions from multiple consecutive (video) frames.



Module Contents
---------------

.. py:function:: sort_faces(lms, img_idx, dist_threshold=250)

   'Sorts' faces across multiple frames.

   :param lms: A float tensor of shape *B* (batch size) x *V* (vertices/landmarks) x *C*
               (coordinates), which will be used to sort the faces
   :type lms: torch.tensor
   :param img_idx: An integer tensor with the image index associated with each detection
                   (e.g., [0, 0, 1, 1, 1, ...] means that there are two faces in the first image,
                   three faces in the second image, etc.)
   :type img_idx: torch.tensor
   :param dist_threshold: Euclidean distance between two sets of landmarks/vertices that we consider
                          comes from two different faces (e.g., if ``d(lms1, lms2) >= dist_treshold``,
                          then we conclude that face 1 (``lms1``) is a different from face 2 (``lms2``)
   :type dist_threshold: torch.tensor

   :returns: **face_idx** -- An integer tensor of length *n detections*, in which each unique value
             represents a unique face
   :rtype: torch.tensor


.. py:function:: filter_faces(face_idx, n_img, present_threshold=0.1)

   Function to filter faces based on various criteria.

   For now, only filters based on how frequent a face is detected
   across frames.
