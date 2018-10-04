#!/usr/local/bin/python
# coding: utf-8
import logging
# ======================================================================
# ---  Setup logging
# ======================================================================
from mdsea.constants.fileman import DIR_SIMFILES

LOG_LEVEL = 20  # logging.INFO
LOGFILE_MAXBYTES = 500000  # 500 kB
LOGFILE_NAME = f"{DIR_SIMFILES}/_mdsea_logdump.log"
LOG_FORMAT_STOUT = "[%(levelname)s:%(name)s] %(message)s"
LOG_FORMAT_FILE = "%(asctime)s - [%(levelname)s:%(name)s] %(message)s"
SUPRESSED_MODULES = ('matplotlib', 'apptools', 'asyncio', 'mayavi', 'pyface')


def _setup_logging(format_file, format_stout, level, filename,
                   maxbytes, supress):
    import logging
    import os
    from logging.handlers import RotatingFileHandler
    
    # Supress logging from other libraries by
    # setting their logging level to CRITICAL
    # TODO: find better way to suppress loggers != mdsea.*
    for log in supress:
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
                            LOGFILE_MAXBYTES, SUPRESSED_MODULES)

# Clean up
del DIR_SIMFILES, LOG_LEVEL, LOG_FORMAT_STOUT, LOGFILE_NAME, LOGFILE_MAXBYTES

# ======================================================================
# ---  Imports
# ======================================================================
from mdsea import vis
from mdsea.analytics import Analyser, make_mp4
from mdsea.core import SysManager
from mdsea.gen import CoordGen, VelGen
from mdsea.potentials import Potential
from mdsea.simulator import ContinuousPotentialSolver
