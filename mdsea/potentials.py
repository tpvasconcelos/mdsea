#!/usr/local/bin/python
# coding: utf-8
import logging
from typing import Callable, Optional

import numpy as np
from scipy import optimize

from mdsea import loghandler

log = logging.getLogger(__name__)
log.addHandler(loghandler)


#######################################
# Potentials (pf: potential function) #
#######################################


def pf_boundedmie(r, a, epsilon, sigma, m, n):
    """ Bounded Mie potential function.  """
    lamb = (m / (m - n)) * (m / n) ** (n / (m - n))
    term = sigma / np.sqrt(a * a + r * r)
    return lamb * epsilon * (term ** m - term ** n)


def pf_mie(r, epsilon, sigma, m, n):
    """ Mie potential function.  """
    return pf_boundedmie(r, a=0, epsilon=epsilon, sigma=sigma, m=m, n=n)


def pf_lennardjones(r, epsilon, sigma):
    """ Lennard Jones potential function.  """
    return pf_mie(r, epsilon=epsilon, sigma=sigma, m=12, n=6)


# noinspection PyUnusedLocal
def pf_ideal(r, **kwargs):
    """ Ideal Gas potential function. """
    return 0


###############################
# Forces (ff: force function) #
###############################


def ff_boundedmie(r, a, epsilon, sigma, m, n):
    """ Bounded Mie Potential potential force function.  """
    lamb = (m / (m - n)) * (m / n) ** (n / (m - n))
    aprs = a * a + r * r
    term = sigma / np.sqrt(aprs)
    return lamb * epsilon * r * ((n * term ** n) - (m * term ** m)) / aprs


def ff_mie(r, epsilon, sigma, m, n):
    """ Mie Potential potential force function.  """
    return ff_boundedmie(r, a=0, epsilon=epsilon, sigma=sigma, m=m, n=n)


def ff_lennardjones(r, epsilon, sigma):
    """ Lennard Jones potential force function. """
    return ff_mie(r, epsilon=epsilon, sigma=sigma, m=12, n=6)


# noinspection PyUnusedLocal
def ff_ideal(r, **kwargs):
    """ Ideal Gas potential force function. """
    return 0


# ----------------------------------------------------------------------


class Potential(object):
    def __init__(self,
                 force: Optional[Callable],
                 potential: Optional[Callable],
                 step: bool = False,
                 kwargs: Optional[dict] = None) -> None:
        """
        If no arguments are passed, this will
         e an ideal gas potential by default.
        
        :param force: potential force function.
        :param potential: potential function.
        :param step: Is it a step (discontinuous) potential function?
        :param kwargs: kwargs for the potential and force functions.
        
        """
        
        # Force function (Callable)
        self.ff = force
        
        # Potential function (Callable)
        self.pf = potential
        
        # Is it a step-potential?
        self.step = step
        
        # kwargs stuff
        self.kwargs = kwargs
        if self.kwargs is None:
            self.kwargs = dict()
    
    # ==================================================================
    # ---  OOTB Potentials (class methods)
    # ==================================================================
    
    @classmethod
    def ideal(cls) -> 'Potential':
        return cls(force=ff_ideal, potential=pf_ideal)
    
    @classmethod
    def boundedmie(cls, a, epsilon, sigma, m, n) -> 'Potential':
        kwargs = dict(a=a, epsilon=epsilon, sigma=sigma, m=m, n=n)
        return cls(force=ff_boundedmie, potential=pf_boundedmie, kwargs=kwargs)
    
    @classmethod
    def mie(cls, epsilon, sigma, m, n) -> 'Potential':
        kwargs = dict(epsilon=epsilon, sigma=sigma, m=m, n=n)
        return cls(force=ff_mie, potential=pf_mie, kwargs=kwargs)
    
    @classmethod
    def lennardjones(cls, epsilon, sigma) -> 'Potential':
        kwargs = dict(epsilon=epsilon, sigma=sigma)
        return cls(force=ff_lennardjones, potential=pf_lennardjones,
                   kwargs=kwargs)
    
    # ==================================================================
    # ---  Public methods
    # ==================================================================
    
    def potential(self, r):
        """ Evaluate and return the potential function at r. """
        return self.pf(r, **self.kwargs)
    
    def force(self, r):
        """ Evaluate and return the force function at r. """
        return self.ff(r, **self.kwargs)
    
    def potminimum(self, xtol: float = 1e-8):
        """
        Get potential minimum / Equilibrium distance / Force's zero
        
        Parameters (from scipy.optimize.fmin)
        ----------
        xtol : float, optional
            Absolute error in xopt between iterations that is acceptable
            for convergence.
        
        """
        return optimize.fmin(self.pf, 0.5, tuple(self.kwargs.values()),
                             xtol=xtol, disp=False)[0]
