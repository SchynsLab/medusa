:py:mod:`medusa.log`
====================

.. py:module:: medusa.log

.. autoapi-nested-parse::

   A module with logging functionality.



Module Contents
---------------

.. py:function:: get_logger(level='INFO')

   Creates a Python logger with a nice format.

   :param level: Logging level ("INFO", "DEBUG", "WARNING")
   :type level: str

   :returns: **logger** -- A Python logger
   :rtype: logging.Logger

   .. rubric:: Examples

   >>> logger = get_logger()
   >>> logger.info("Hello!")


.. py:function:: tqdm_log(iter_, logger, desc='Render shape')

   Creates an iterator with optional tqdm progressbar that plays nicely
   with an existing Medusa logger.

   :param iter_: Initial iterable
   :type iter_: iterable
   :param logger: Existing Medusa logger
   :type logger: logging.Logger
   :param desc: Text to display before progress bar
   :type desc: str

   :returns: **iter_** -- Iterable to be iterated over
   :rtype: iterable
