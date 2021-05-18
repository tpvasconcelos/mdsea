"""
mdsea - Molecular Dynamics Library

mdsea is a stand-alone Python molecular dynamics library equipped with
a flexible simulation engine and multiple analysis tools, including
integrated beautiful visualization in 1, 2, and 3 dimensions.

"""
from mdsea._version import __version__
from mdsea.analytics import Analyser, make_mp4
from mdsea.core import SysManager
from mdsea.gen import PosGen, VelGen
from mdsea.potentials import Potential
from mdsea.simulator import ContinuousPotentialSolver

__all__ = [
    "Analyser",
    "make_mp4",
    "SysManager",
    "PosGen",
    "VelGen",
    "Potential",
    "ContinuousPotentialSolver",
    "__version__",
]
