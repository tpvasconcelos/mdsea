"""
Remove the entire simulation directory.

"""
from mdsea.core import SysManager
from mdsea.helpers import setup_logging

setup_logging(level="DEBUG")

sm = SysManager.load(simid="_mdsea_docs_example")
sm.delete()
