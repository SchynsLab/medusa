:orphan:

:py:mod:`medusa.recon.mpipe._transforms`
========================================

.. py:module:: medusa.recon.mpipe._transforms

.. autoapi-nested-parse::

   A Python implementation of the C++ transform module in mediapipe by Rasmus
   Jones (https://github.com/Rassibassi), adapted from `here. <https://github.com/Rassibassi/mediapipeDemos/blob/main/head_posture.py>`__ and
   `here <https://github.com/Rassibassi/mediapipeDemos/blob/main/custom/face_geometry.py>`__.

   The code in the module is used to estimate the local-to-world matrix of the ``Mediapipe``
   reconstruction, i.e., how the current reconstruction is transformed relative to the
   canonical Mediapipe model (which is available
   `here <https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/data>`__).

   The class/functions are not documented as they are unlikely to be actually used by
   users of Medusa.



Module Contents
---------------

.. py:class:: PCF(near=1, far=10000, frame_height=1920, frame_width=1080, fy=1080)


.. py:data:: procrustes_landmark_basis
   :value: [(4, 0.070909939706326), (6, 0.032100144773722), (10, 0.008446550928056), (33,...

   

.. py:data:: landmark_weights

   

.. py:function:: image2world(screen_landmarks, pcf, v_world_ref)


.. py:function:: project_xy(landmarks, pcf)


.. py:function:: change_handedness(landmarks)


.. py:function:: move_and_rescale_z(pcf, depth_offset, scale, landmarks)


.. py:function:: unproject_xy(pcf, landmarks)


.. py:function:: estimate_scale(landmarks, v_world_ref)


.. py:function:: solve_weighted_orthogonal_problem(source_points, target_points, point_weights)


.. py:function:: internal_solve_weighted_orthogonal_problem(sources, targets, sqrt_weights)


.. py:function:: compute_optimal_rotation(design_matrix)


.. py:function:: compute_optimal_scale(centered_weighted_sources, weighted_sources, weighted_targets, rotation)


.. py:function:: combine_transform_matrix(r_and_s, t)


