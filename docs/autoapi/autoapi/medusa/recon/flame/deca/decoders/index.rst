:py:mod:`medusa.recon.flame.deca.decoders`
==========================================

.. py:module:: medusa.recon.flame.deca.decoders

.. autoapi-nested-parse::

   Decoders specific to DECA-based reconstruction models.

   For the associated license, see license.md.



Module Contents
---------------

.. py:class:: DetailGenerator(latent_dim=100, out_channels=1, out_scale=0.01, sample_mode='bilinear')



   A generator that converts latents into a detail map.

   .. py:method:: forward(noise)

      Forward pass.
