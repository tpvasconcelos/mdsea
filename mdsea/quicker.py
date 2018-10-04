#!/usr/local/bin/python
# coding: utf-8
import numpy as np
# noinspection PyProtectedMember
from scipy.spatial.distance import _distance_wrap

from mdsea.constants import DTYPE


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
    id_.flat[::n + 1] = 1
    return id_[(slice(None, None, None), slice(None, None, -1))]


def norm(x, axis=None):
    return np.sqrt(np.add.reduce(x ** 2, axis=axis, dtype=DTYPE), dtype=DTYPE)
