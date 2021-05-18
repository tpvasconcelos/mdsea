"""
This is a list of optimized functions with the goal of speeding up the
simulation. Some of these functions are based on existing ones from
packages like numpy and scipy, but since we know the input of these
functions we can skip all the checks and extra steps performed by
such packages.

"""

import numpy as np
from mdsea.constants import DTYPE
from scipy.spatial.distance import _distance_wrap


def cdist_(x: np.ndarray) -> np.ndarray:
    """
    Fast C wrapper for spacial distance calculations.

    The real function is 'cdist' from 'scipy.spatial.distance'.

    This function skips a all the checks
    performed by scipy.spatial.distance

    Equivalent to (but without all the checks):

    >>> from scipy.spatial.distance import cdist
    >>> cdist(x, x)

    """
    m = x.shape[0]
    dm = np.empty((m, m), dtype=np.double)
    _distance_wrap.cdist_euclidean_double_wrap(x, x, dm)
    # assert np.nan_to_num(dm) == dm, 'np.nan found in dm'
    return np.nan_to_num(dm)


def flipid(n):
    """ Flipped identity matrix. """
    id_ = np.zeros((n, n), dtype=DTYPE)
    id_.flat[:: n + 1] = 1
    return id_[(slice(None, None, None), slice(None, None, -1))]


def norm(x, axis=None):
    """ Returns the norm of x. """
    return np.sqrt(np.add.reduce(x ** 2, axis=axis, dtype=DTYPE), dtype=DTYPE)
