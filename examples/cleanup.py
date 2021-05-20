"""
Remove the entire simulation directory.

"""
import mdsea as md
from mdsea.helpers import setup_logging

setup_logging(level="DEBUG")

sm = md.SysManager.load(simid="_mdsea_docs_example")
sm.delete()
