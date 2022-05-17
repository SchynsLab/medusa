:py:mod:`medusa.utils`
======================

.. py:module:: medusa.utils


Module Contents
---------------

.. py:function:: get_logger(verbose='INFO')

   Create a Python logger.

   :param verbose: Logging level ("INFO", "DEBUG", "WARNING")
   :type verbose: str

   :returns: **logger** -- A Python logger
   :rtype: logging.Logger

   .. rubric:: Examples

   >>> logger = get_logger()
   >>> logger.info("Hello!")


