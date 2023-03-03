:py:mod:`medusa.recon.flame.lbs`
================================

.. py:module:: medusa.recon.flame.lbs

.. autoapi-nested-parse::

   Module with functionality for linear blend skinning.

   See ./deca/license.md for conditions for use.



Module Contents
---------------

.. py:function:: rot_mat_to_euler(rot_mats)


.. py:function:: lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot=True)

   Performs Linear Blend Skinning with the given shape and pose parameters.

   :param betas: The tensor of shape parameters
   :type betas: torch.tensor BxNB
   :param pose: The pose parameters in axis-angle format
   :type pose: torch.tensor Bx(J + 1) * 3
   :param v_template torch.tensor BxVx3: The template mesh that will be deformed
   :param shapedirs: The tensor of PCA shape displacements
   :type shapedirs: torch.tensor 1xNB
   :param posedirs: The pose PCA coefficients
   :type posedirs: torch.tensor Px(V * 3)
   :param J_regressor: The regressor array that is used to calculate the joints from
                       the position of the vertices
   :type J_regressor: torch.tensor JxV
   :param parents: The array that describes the kinematic tree for the model
   :type parents: torch.tensor J
   :param lbs_weights: The linear blend skinning weights that represent how much the
                       rotation matrix of each part affects each vertex
   :type lbs_weights: torch.tensor N x V x (J + 1)
   :param pose2rot: Flag on whether to convert the input pose tensor to rotation
                    matrices. The default value is True. If False, then the pose tensor
                    should already contain rotation matrices and have a size of
                    Bx(J + 1)x9
   :type pose2rot: bool, optional
   :param dtype:
   :type dtype: torch.dtype, optional

   :returns: * **verts** (*torch.tensor BxVx3*) -- The vertices of the mesh after applying the shape and pose
               displacements.
             * **joints** (*torch.tensor BxJx3*) -- The joints of the model


.. py:function:: blend_shapes(betas, shape_disps)

   Calculates the per vertex displacement due to the blend shapes.

   :param betas: Blend shape coefficients
   :type betas: torch.tensor Bx(num_betas)
   :param shape_disps: Blend shapes
   :type shape_disps: torch.tensor Vx3x(num_betas)

   :returns: The per-vertex displacement due to shape deformation
   :rtype: torch.tensor BxVx3


.. py:function:: batch_rodrigues(rot_vecs, epsilon=1e-08)

   Calculates the rotation matrices for a batch of rotation vectors
   :param rot_vecs: array of N axis-angle vectors
   :type rot_vecs: torch.tensor Nx3

   :returns: **R** -- The rotation matrices for the given axis-angle parameters
   :rtype: torch.tensor Nx3x3


.. py:function:: transform_mat(R, t)

   Creates a batch of transformation matrices
   :param - R: Bx3x3 array of a batch of rotation matrices
   :param - t: Bx3x1 array of a batch of translation vectors

   :returns: Bx4x4 Transformation matrix
   :rtype: - T


.. py:function:: batch_rigid_transform(rot_mats, joints, parents)

   Applies a batch of rigid transformations to the joints.

   :param rot_mats: Tensor of rotation matrices
   :type rot_mats: torch.tensor BxNx3x3
   :param joints: Locations of joints
   :type joints: torch.tensor BxNx3
   :param parents: The kinematic tree of each object
   :type parents: torch.tensor BxN
   :param dtype: The data type of the created tensors, the default is torch.float32
   :type dtype: torch.dtype, optional:

   :returns: * **posed_joints** (*torch.tensor BxNx3*) -- The locations of the joints after applying the pose rotations
             * **rel_transforms** (*torch.tensor BxNx4x4*) -- The relative (with respect to the root joint) rigid transformations
               for all the joints
