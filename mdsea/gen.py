import logging
import math

import numpy as np
from mdsea.constants import DTYPE
from scipy import special
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)

MONTECARLO_SPEEDRANGE = np.arange(0, 50, 0.001)


# ======================================================================
# ---  Speed Distributions
# ======================================================================


def mb(mass, temp, k_boltzmann) -> np.ndarray:
    """ Returns Maxwell-Boltzmann's speed distribution. """
    term1 = mass / (2 * math.pi * k_boltzmann * temp)
    term2 = 4 * math.pi * MONTECARLO_SPEEDRANGE ** 2
    term3 = -mass * MONTECARLO_SPEEDRANGE ** 2 / (2 * k_boltzmann * temp)
    return term1 ** 1.5 * term2 * np.exp(term3)


def mb_cdf(mass, temp, k_boltzmann) -> np.ndarray:
    """
    Returns a Maxwell-Boltzmann's speed distribution
    Cumulative Distribution Function (CDF).

    """
    term0 = math.sqrt(k_boltzmann * temp / mass)
    term1 = special.erf(MONTECARLO_SPEEDRANGE / (math.sqrt(2) * term0))
    term2 = np.sqrt(2 / math.pi) * MONTECARLO_SPEEDRANGE
    term3 = np.exp(-(MONTECARLO_SPEEDRANGE ** 2) / (2 * term0 ** 2)) / term0
    return term1 - term2 * term3


# ======================================================================
# ---  Classes
# ======================================================================


class _Gen:
    def __init__(self, nparticles: int, ndim: int) -> None:
        self.nparticles = nparticles
        self.ndim = ndim

        # This will change to True once a
        # 'generator' is successfully called
        self.generated = False

        # This will be updated by a 'generator'
        self.coords = np.zeros((self.ndim, self.nparticles), dtype=DTYPE)

        # ==================================================================
        # ---  Private methods
        # ==================================================================

    def _get(self) -> np.ndarray:
        """
        Returns numpy ndarray of coordinates
        in their dimensional components (x1, x2, x3, ...).

        """
        if not self.generated:
            log.warning("Coordinates haven't been generated yet!")
        return self.coords.astype(DTYPE)


class VelGen(_Gen):
    def __init__(self, nparticles: int, ndim: int) -> None:

        super().__init__(nparticles, ndim)

        self.speeds = None

    # ==================================================================
    # ---  Private methods: Cartesian Coordinates
    # ==================================================================

    def _mb_coords(self, i, r, thetas):
        if i == self.ndim - 1:
            cc = r * np.cos(thetas[self.ndim - 2]) * np.prod([np.sin(thetas[j]) for j in range(self.ndim - 2)])
        elif i == self.ndim - 2:
            cc = r * np.sin(thetas[self.ndim - 2]) * np.prod([np.sin(thetas[j]) for j in range(self.ndim - 2)])
        elif i == 0:
            cc = r * np.cos(thetas[0])
        else:
            cc = r * np.cos(thetas[i]) * np.prod([np.sin(thetas[j]) for j in range(i)])
        return cc

    # ==================================================================
    # ---  Public/User methods: Generators
    # ==================================================================

    def mb(self, mass, temp, k_boltzmann) -> np.ndarray:
        """ Generate Maxwell-Boltzmann velocities. """

        # Special case for T=0  ---
        if temp == 0:
            self.coords = np.zeros((self.ndim, self.nparticles), dtype=DTYPE)
            self.generated = True
            return self._get()

        # Generate a MB distribution of speeds  ---
        # Note: interp1d is a class
        inv_cdf = interp1d(mb_cdf(mass, temp, k_boltzmann), MONTECARLO_SPEEDRANGE)
        self.speeds = inv_cdf(np.random.random(self.nparticles))

        # Generate spherical coordinates (range)  ---
        # One of the coordinates has range [0, 2*pi],
        # the others (ndim - 1) have range [0, pi]
        thetas = [np.random.uniform(0, math.pi, self.nparticles) for _ in range(self.ndim - 2)]
        thetas.append(np.random.uniform(0, 2 * math.pi, self.nparticles))

        # Get the cartesian coordinates  ---
        self.coords = np.array([self._mb_coords(i, self.speeds, thetas) for i in range(self.ndim)])

        # FINISH  ---
        self.generated = True
        return self._get()


class PosGen(_Gen):
    def __init__(self, nparticles: int, ndim: int, boxlen: float) -> None:

        super().__init__(nparticles, ndim)

        self.boxlen = boxlen

    # ==================================================================
    # ---  Private methods: Cartesian Coordinates
    # ==================================================================

    # def _foo_coords(self, ...):
    #     return

    # ==================================================================
    # ---  Public/User methods: Generators
    # ==================================================================

    def simplecubic(self) -> np.ndarray:
        """ Return simple cubic coordinates. """
        # particles per row  ---
        ppr = round(self.nparticles ** (1 / self.ndim))
        assert self.nparticles == ppr ** self.ndim
        ppr_range = 0.5 + np.fromiter(list(range(ppr)), dtype=int)
        # particle span  ---
        pspan = self.boxlen / ppr
        for i in range(self.ndim):
            self.coords[i] = pspan * np.tile(ppr_range.repeat(ppr ** i), ppr ** (self.ndim - i - 1))
        # FINISH  ---
        self.generated = True
        return self._get()

    def random(self, pradius) -> np.ndarray:
        """ Return random coordinates. """
        # Usable box span  ---
        us = self.boxlen - 2 * pradius
        # Distribute coordinates randomly  ---
        self.coords = us * np.random.random((self.ndim, self.nparticles))
        # Set/add positive offset  ---
        self.coords += pradius
        # FINISH  ---
        self.generated = True
        return self._get()
