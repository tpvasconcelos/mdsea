#!/usr/local/bin/python
# coding: utf-8

"""
mdsea - Molecular Dynamics Library

mdsea is a stand-alone Python molecular dynamics library equipped with
a flexible simulation engine and multiple analysis tools, including
integrated beautiful visualization in 1, 2, and 3 dimensions.

In this file we start by setting up the logging standards and format
used in this package. The RotatingFileHandler (FileHandler) instance
is used throughout the whole package. For consistency, we handle this
on the top of the module right after imports. Do as follows:

>>> import logging
>>> from mdsea import loghandler
>>>
>>> log = logging.getLogger(__name__)
>>> log.addHandler(loghandler)

"""

# ======================================================================
# ---  Setup logging
# ======================================================================
from mdsea.constants.fileman import DIR_SIMFILES

LOG_LEVEL = 20  # logging.INFO
LOGFILE_MAXBYTES = 500000  # 500 kB
LOGFILE_NAME = f"{DIR_SIMFILES}/_mdsea_logdump.log"
LOG_FORMAT_STOUT = "[%(levelname)s:%(name)s] %(message)s"
LOG_FORMAT_FILE = "%(asctime)s - [%(levelname)s:%(name)s] %(message)s"
SUPPRESSED_MODULES = ('matplotlib',
                      'apptools',
                      'asyncio',
                      'mayavi',
                      'pyface')


def _setup_logging(format_file, format_stout, level, filename,
                   maxbytes, suppress) -> 'RotatingFileHandler':
    """ Setup mdsea's logging. """
    import logging
    import os
    from logging.handlers import RotatingFileHandler
    
    # Suppress logging from other libraries by
    # setting their logging level to CRITICAL
    # TODO(tpvasconcelos) find better way to suppress loggers != mdsea.*
    for log in suppress:
        logging.getLogger(log).setLevel(logging.CRITICAL)
    
    # Basic config
    logging.basicConfig(format=format_stout, level=level)
    
    # Check if the directory where the log will live exists
    _dir = "/".join(filename.split('/')[:-1])
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    
    # Set handler
    formatter = logging.Formatter(format_file)
    handler = RotatingFileHandler(filename, maxBytes=maxbytes, backupCount=1)
    handler.setFormatter(formatter)
    
    return handler


loghandler = _setup_logging(LOG_FORMAT_FILE, LOG_FORMAT_STOUT,
                            LOG_LEVEL, LOGFILE_NAME,
                            LOGFILE_MAXBYTES, SUPPRESSED_MODULES)

# Clean up
del DIR_SIMFILES, LOG_LEVEL, LOG_FORMAT_STOUT, LOGFILE_NAME, LOGFILE_MAXBYTES

# ======================================================================
# ---  Imports
# ======================================================================
from mdsea import vis
from mdsea.analytics import Analyser, make_mp4
from mdsea.core import SysManager
from mdsea.gen import PosGen, VelGen
from mdsea.potentials import Potential
from mdsea.simulator import ContinuousPotentialSolver
