#!/usr/local/bin/python
# coding: utf-8
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import special
from scipy.interpolate import interp1d

from mdsea.constants import DTYPE

if TYPE_CHECKING:
    ############################
    #  Fix for cyclic imports  #
    ############################
    # Note that the 'SysManager' type annotation is turned into
    # a string. This is needed since SysManager won't be
    # available at runtime when TYPE_CHECKING == False
    pass

from mdsea import loghandler

log = logging.getLogger(__name__)
log.addHandler(loghandler)

MONTECARLO_SPEEDRANGE = np.arange(0, 50, 0.001)


# ======================================================================
# ---  Speed Distributions
# ======================================================================


def mb(mass, temp, k_boltzmann) -> np.ndarray:
    """ Returns Maxwell-Boltzmann's speed distribution. """
    term1 = mass / (2 * math.pi * k_boltzmann * temp)
    term2 = 4 * math.pi * MONTECARLO_SPEEDRANGE ** 2
    term3 = -mass * MONTECARLO_SPEEDRANGE ** 2 / (2 * k_boltzmann * temp)
    return (term1 ** 1.5) * term2 * np.exp(term3)


def mb_cdf(mass, temp, k_boltzmann) -> np.ndarray:
    """
    Returns a Maxwell-Boltzmann's speed distribution
    Cumulative Distribution Function (CDF).

    """
    term0 = math.sqrt(k_boltzmann * temp / mass)
    term1 = special.erf(MONTECARLO_SPEEDRANGE / (math.sqrt(2) * term0))
    term2 = np.sqrt(2 / math.pi) * MONTECARLO_SPEEDRANGE
    term3 = np.exp(-MONTECARLO_SPEEDRANGE ** 2 / (2 * term0 ** 2)) / term0
    return term1 - term2 * term3


# ======================================================================
# ---  Classes
# ======================================================================


class _Gen:
    def __init__(self, nparticles: int, ndim: int) -> None:
        self.nparticles = nparticles
        self.ndim = ndim
        
        # This will change to True once a
        # 'generator' is susscefully called
        self.generated = False
        
        # This will be updated by a 'generator'
        self.speeds = None
        self.coords = np.zeros((self.ndim, self.nparticles), dtype=DTYPE)
        
        # ==================================================================
        # ---  Private methods
        # ==================================================================
    
    def _get(self) -> np.ndarray:
        """
        Returns numpy ndarray of coordinates
        in their dimentional components (x1, x2, x3, ...).

        """
        if not self.generated:
            log.warning("Coordinates haven't been generated yet!")
        return self.coords.astype(DTYPE)


class VelGen(_Gen):
    def __init__(self, nparticles: int, ndim: int) -> None:
        
        super(VelGen, self).__init__(nparticles, ndim)
    
    # ==================================================================
    # ---  Private methods: Cartesian Coordinates
    # ==================================================================
    
    def _mb_coords(self, i, r, thetas):
        cc = r
        if i == self.ndim - 1:
            cc *= np.cos(thetas[self.ndim - 2]) \
                  * np.prod([np.sin(thetas[j]) for j in range(self.ndim - 2)])
        elif i == self.ndim - 2:
            cc *= np.sin(thetas[self.ndim - 2]) \
                  * np.prod([np.sin(thetas[j]) for j in range(self.ndim - 2)])
        elif i == 0:
            cc *= np.cos(thetas[0])
        else:
            cc *= np.cos(thetas[i]) \
                  * np.prod([np.sin(thetas[j]) for j in range(i)])
        return cc
    
    # ==================================================================
    # ---  Public/User methods: Generators
    # ==================================================================
    
    def mb(self, mass, temp, k_boltzmann) -> np.ndarray:
        """ Generate Maxwell-Boltzmann velocites. """
        
        if temp == 0:
            self.coords = np.zeros((self.ndim, self.nparticles), dtype=DTYPE)
            self.generated = True
            return self._get()
        
        # Note: interp1d is a class
        inv_cdf = interp1d(mb_cdf(mass, temp, k_boltzmann),
                           MONTECARLO_SPEEDRANGE)
        self.speeds = inv_cdf(np.random.random(self.nparticles))
        
        thetas = [np.random.uniform(0, math.pi, self.nparticles)
                  for _ in range(self.ndim - 2)]
        # if self.ndim > 1:
        thetas.append(np.random.uniform(0, 2 * math.pi, self.nparticles))
        
        self.coords = np.array([self._mb_coords(i, self.speeds, thetas)
                                for i in range(0, self.ndim)])
        
        self.generated = True
        return self._get()


class CoordGen(_Gen):
    def __init__(self, nparticles: int, ndim: int,
                 boxlen: float) -> None:
        
        super(CoordGen, self).__init__(nparticles, ndim)
        
        self.boxlen = boxlen
    
    # ==================================================================
    # ---  Public/User methods: Generators
    # ==================================================================
    
    def simplecubic(self) -> np.ndarray:
        
        def coord_i(i):
            f_rep = ppr ** i
            f_til = ppr ** (self.ndim - (i + 1))
            return pspan * np.tile(ppr_range.repeat(f_rep), f_til)
        
        # particles per row  ---
        ppr = round(self.nparticles ** (1 / self.ndim))
        assert self.nparticles == ppr ** self.ndim
        ppr_range = 0.5 + np.fromiter(list(range(ppr)), dtype=int)
        # particle span  ---
        pspan = self.boxlen / ppr
        
        for n in range(self.ndim):
            self.coords[n] = coord_i(n)
        
        self.generated = True
        return self._get()
    
    def random(self, pradius) -> np.ndarray:
        
        # Usable box span
        us = self.boxlen - 2 * pradius
        
        self.coords = us * np.random.random((self.ndim, self.nparticles)) \
                      + pradius
        
        self.generated = True
        return self._get()
