:py:mod:`medusa.defaults`
=========================

.. py:module:: medusa.defaults

.. autoapi-nested-parse::

   Module with default objects, which values may depend on whether the system
   has access to a GPU or not (such as ``DEVICE``).



Module Contents
---------------

.. py:data:: DEVICE

   Default device ('cuda' or 'cpu') used across Medusa, which depends on whether
   *cuda* is available ('cuda') or not ('cpu').

.. py:data:: FONT

   Default font used in Medusa (DejaVuSans).

.. py:data:: FLAME_MODELS
   :value: ['deca-coarse', 'deca-dense', 'emoca-coarse', 'emoca-dense', 'spectre-coarse', 'spectre-dense']

   Names of available FLAME-based models, which can be used when initializing a
   ``DecaReconModel``.

.. py:data:: RECON_MODELS
   :value: ['spectre-coarse', 'emoca-dense', 'emoca-coarse', 'deca-dense', 'deca-coarse', 'mediapipe']

   Names of all available reconstruction models.

.. py:data:: LOGGER

   Default logger used in Medusa.

.. py:data:: default_renderer



.. py:data:: RENDERER

   Default renderer used in Medusa, which depends on whether ``pytorch3d`` is installed
   (in which case ``PytorchRenderer`` is used) or not (no rendering possible).
