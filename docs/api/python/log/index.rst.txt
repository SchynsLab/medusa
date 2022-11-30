:py:mod:`medusa.log`
====================

.. py:module:: medusa.log


Module Contents
---------------

.. py:function:: get_logger(level='INFO')

   Create a Python logger.

   :param level: Logging level ("INFO", "DEBUG", "WARNING")
   :type level: str

   :returns: **logger** -- A Python logger
   :rtype: logging.Logger

   .. rubric:: Examples

   >>> logger = get_logger()
   >>> logger.info("Hello!")


