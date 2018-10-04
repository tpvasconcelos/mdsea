#!/usr/local/bin/python
# coding: utf-8
import logging

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from mdsea import loghandler
from mdsea.gen import MONTECARLO_SPEEDRANGE, VelGen
from mdsea.potentials import pf_boundedmie

log = logging.getLogger(__name__)
log.addHandler(loghandler)

BIG_FONT = 25
SMALL_FONT = 18


def _color(val, minimum, maximum):
    if val == 1.12246:
        val = 1.08
    speed_ratio = round(10000 * (val - minimum) / (maximum - minimum))
    # gist_heat jet Spectral_r gnuplot brg coolwarm autumn copper
    cmap_name = 'coolwarm'
    cmap = cm.get_cmap(cmap_name, 10000)
    return cmap(speed_ratio)


def plot_continuous_potential():
    r = np.arange(0, 3, 0.001)
    phi = pf_boundedmie(r)
    plt.plot(r, phi)
    plt.show()
    plt.cla()
    plt.clf()


def plot_sclj():
    x = np.linspace(0, 3, 500)
    # a_range = [0, 0.5, 0.7, 0.88]
    a_range = np.linspace(1.12246, 2.2, 800)
    # a_range = [0.88, 0.96, 1, 1.025, 1.12246]
    # a_range = [1.12246, 1.3, 1.6, 2.2]
    a_max = np.max(a_range)
    a_min = np.min(a_range)
    
    # fig = plt.figure()
    # ax = fig.add_axes([0.12, 0.15, 0.5, 0.8])
    
    for a_param in a_range:
        y = [pf_boundedmie(r, a=a_param) for r in x]
        plt.plot(x, y, color=_color(a_param, a_min, a_max), lw=1,
                 label=r'$a / \sigma = {}$'.format(str(round(a_param, 5))))
    
    y_lim = [-1.1, 0.1]
    plt.ylim(y_lim)
    
    # plt.yticks(range(int(min(y_lim)), int(max(y_lim)) + 1), size=SMALL_FONT)
    plt.yticks([-1, -0.5, 0, 0.5], size=SMALL_FONT)
    plt.xticks(range(int(max(x)) + 1), size=SMALL_FONT)
    
    plt.ylabel(r'$\phi(r) / \epsilon$', size=BIG_FONT)
    plt.xlabel(r'$r / \sigma$', size=BIG_FONT)
    
    plt.grid(which='both')
    # plt.legend(loc='best', prop={'size': BIG_FONT})
    
    my_cb = plt.contourf([[-10, 0], [0, 0]], a_range,
                         cmap=cm.get_cmap('coolwarm'))
    cbar = plt.colorbar(my_cb)
    cbar.set_ticks([min(a_range), max(a_range)])
    cbar.set_ticklabels([r'$a_{\mathrm{m}}$', r'$2.2$'])
    cbar.set_label(r'$a / \sigma$', size=BIG_FONT, rotation=0)
    cbar.ax.yaxis.set_label_coords(3, 0.5)
    cbar.ax.tick_params(labelsize=SMALL_FONT)
    
    plt.tight_layout()
    plt.savefig('my_plot.png', dpi=300)
    
    plt.show()


def plot_sclj_different_mns():
    x = np.arange(0, 10, 0.01)
    
    y = [pf_boundedmie(r, a=0, m=2, n=1) for r in x]
    plt.plot(x, y, lw=3)
    
    y_lim = [-1.25, 3]
    plt.ylim(y_lim)
    
    plt.yticks(range(int(min(y_lim)), int(max(y_lim)) + 1), size=SMALL_FONT)
    # plt.yticks([-1, -0.5, 0, 0.5], size=SMALL_FONT)
    plt.xticks(range(int(max(x)) + 1), size=SMALL_FONT)
    
    plt.ylabel(r'$\phi(r) / \epsilon$', size=BIG_FONT)
    plt.xlabel(r'$r / \sigma$', size=BIG_FONT)
    
    plt.grid(which='both')
    plt.legend(loc='best', prop={'size': BIG_FONT})
    
    plt.tight_layout()
    # plt.savefig('/Users/tomasvasconcelos/Desktop/my_plot.png', dpi=300)
    
    plt.show()


def mb_speed_distribution_example():
    velgen = VelGen()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for temperature in [100, 200, 300, 400]:
        fv = velgen.mb()
        ax.plot(MONTECARLO_SPEEDRANGE, fv,
                label='T=' + str(temperature) + ' K', lw=2)
    ax.legend(loc=0)
    ax.set_xlabel('$speeds_range_monte_carlo$ (m/s)')
    ax.set_ylabel('PDF, $f_v(speeds_range_monte_carlo)$')
    ax.set_xlim(0, 800)
    plt.show()


def mb_cdf_example():
    velgen = VelGen()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for temperature in [100, 200, 300, 400]:
        fv = velgen.mb_cdf()
        ax.plot(MONTECARLO_SPEEDRANGE, fv,
                label='T=' + str(temperature) + ' K', lw=2)
    ax.legend(loc=0)
    ax.set_xlabel('$speeds_range_monte_carlo$ (m/s)')
    ax.set_ylabel('CDF, $C_v(speeds_range_monte_carlo)$')
    ax.set_xlim(0, 800)
    plt.show()


def mb_monte_carlo_speed_distribution_example():
    vgen = VelGen()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # generate histogram of velocities
    ax.hist(vgen.speeds, bins=100, normed=True, fc='b', alpha=0.4, lw=0.2)
    # compare this histogram to f(speeds_range_monte_carlo)
    # this is MB_speed that we wrote earlier
    fv = vgen.mb(1, 1, 1)
    ax.plot(MONTECARLO_SPEEDRANGE, fv, 'k', lw=2)
    ax.set_xlabel(r'Speed $(m/s)$', size=BIG_FONT)
    ax.set_ylabel('PDF', size=BIG_FONT)
    ax.set_xlim(0, 50)
    plt.savefig('my_file24r.png',
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    mb_monte_carlo_speed_distribution_example()
