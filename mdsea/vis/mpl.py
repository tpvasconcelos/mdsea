#!/usr/local/bin/python
# coding: utf-8

"""
matplotlib visualizations and animations.

"""

import logging
from typing import Tuple
from abc import abstractmethod, ABCMeta

import matplotlib
from matplotlib.collections import PathCollection
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.artist import Artist

from mdsea import loghandler
from mdsea.analytics import Vis
from mdsea.core import SysManager
from mdsea.helpers import ProgressBar

log = logging.getLogger(__name__)
log.addHandler(loghandler)


def speed2color(speed: float, speed_limit: float, cmap: Colormap = None,
                alpha: bool = True) -> tuple:
    """ Transform a speed into a rgb (or rgba) color. """
    # TODO: vectorize this s.t. 'speed' can be an array
    if cmap is None:
        # DIVERGING -> ['coolwarm', 'RdBu_r', 'jet']
        # SEQUENTIAL -> ['gist_heat', 'autumn', 'hot']
        num_colors = 256 / 2
        cmap = cm.get_cmap(name='autumn', lut=num_colors)
    # speed_ratio = round(cmap.N * speed / speed_limit)
    speed_ratio = cmap.N - int(cmap.N * speed / speed_limit)
    if cmap.name != 'autumn':
        # Reverse the colors for the 'autumn' colormap.
        speed_ratio = cmap.N - speed_ratio
    if not alpha:
        # Remove alpha (transparency)
        return cmap(speed_ratio)[:-1]
    return cmap(speed_ratio)


class MPL(Vis):
    """ matplotlib visualizations. """
    
    def __init__(self,
                 sm: SysManager,
                 frame_step: int = 1,
                 backend: str = "Qt5Agg",
                 ) -> None:
        super(MPL, self).__init__(sm, frame_step)

        if backend is not None:
            matplotlib.use(backend)

        if self.sm.NDIM == 1:
            zeroes_ = (self.sm.LEN_BOX / 2) + np.zeros((self.sm.STEPS,
                                                        self.sm.NUM_PARTICLES))
            self.r_coords = np.stack([self.r_coords[:, 0], zeroes_], axis=1)
            self.r_vecs = np.stack([self.r_coords[:, 0], zeroes_], axis=-1)
        
        # FIXME(tpvasconcelos) dark_theme @property doesn't work!!
        self._dark_theme = False
    
    @property
    def dark_theme(self):
        """ Set mpl theme. """
        return self._dark_theme
    
    @dark_theme.setter
    def dark_theme(self, val):
        if not isinstance(val, bool):
            raise TypeError("'val' has to be a bool.")
        self._dark_theme = val
        if self._dark_theme:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
    
    def plt_energies(self, lw: float = 2, font_size: float = 18,
                     label_pe: str = 'Potential Energy',
                     label_ke: str = 'Kinetic Energy',
                     label_te: str = 'Total Energy') -> None:
        """ Plot energies over time. """
        plt.plot(self.mean_pot_energies, lw=lw, label=label_pe)
        plt.plot(self.mean_kin_energies, lw=lw, label=label_ke)
        plt.plot(self.total_energy, lw=lw, label=label_te)
        plt.legend(loc='best', prop={'size': font_size})
        plt.grid()
        self._safe_show(self.plt_energies.__name__)
    
    def plt_temp(self,
                 lw: float = 2,
                 fontsize: float = 18,
                 label: str = r'$T^*$') -> None:
        """ Plot temperature over time. """
        plt.plot(self.temp, lw=lw, label=label)
        plt.legend(loc='best', prop={'size': fontsize})
        plt.grid()
        self._safe_show(self.plt_temp.__name__)
    
    def plt_sd(self,
               fontsize: float = 18,
               label: str = r'$Speed Distribution$') -> None:
        """ Plot the speed distribution. """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # generate histogram of velocities
        ax.hist(self.speeds.flatten(), bins=100, density=True, label=label)
        # compare this histogram to f(speeds_range_monte_carlo)
        # this is MB_speed that we wrote earlier
        ax.set_xlabel(r'Speed $(m/s)$', size=fontsize)
        ax.set_ylabel('PDF', size=fontsize)
        plt.legend(loc='best', prop={'size': fontsize})
        plt.grid()
        self._safe_show(self.plt_sd.__name__)
    
    def plt_rdf(self,
                lw: float = 2,
                fontsize: float = 18,
                label: str = r'$g(r)$') -> None:
        """ Plot the Radial Distribution Function (RDF). """
        plt.plot(self.temp, lw=lw, label=label)
        plt.legend(loc='best', prop={'size': fontsize})
        plt.grid()
        self._safe_show(self.plt_rdf.__name__)
    
    @staticmethod
    def _safe_show(name: str) -> None:
        log.info(f'Plotting {name}...')
        try:
            return plt.show()
        finally:
            plt.close('all')


class _BaseMLPAnimation(MPL, metaclass=ABCMeta):
    """Base matplotlib animation."""

    def __init__(self,
                 # MPL kwargs:
                 sm: SysManager,
                 frame_step: int = 1,
                 backend: str = "Qt5Agg",
                 # Animation kwargs:
                 colorspeed=False,
                 color: str = 'orange',
                 figsize: Tuple[int, int] = (15, 10),
                 ) -> None:

        super(_BaseMLPAnimation, self).__init__(sm, frame_step, backend)

        self.dflt_clr = color
        self.colorspeed = colorspeed

        if self.colorspeed:
            self.colors = [[speed2color(speed=s, speed_limit=self.maxspeed) for s in ss] for ss in self.speeds]

        self.fig, self.ax = plt.subplots(figsize=figsize)

        # axis preferences
        min_ = -0.1
        max_ = self.sm.LEN_BOX + 0.1
        self.ax.axis(xmin=min_, xmax=max_, ymin=min_, ymax=max_)
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()

        # Draw the box where the particles are contained in
        if self.sm.NDIM == 1:
            height = 2 * self.sm.RADIUS_PARTICLE
            origin = (0, self.sm.LEN_BOX / 2 - height / 2)
        else:
            height = self.sm.LEN_BOX
            origin = (0, 0)
        self.ax.add_patch(Rectangle(xy=origin, width=self.sm.LEN_BOX, height=height, lw=1, fill=False))

        self.pbarr = ProgressBar("Saving animation frame", self.sm.STEPS, __name__)

    @property
    @abstractmethod
    def artists(self) -> Tuple[Artist, ...]:
        pass

    @abstractmethod
    def _add_particles(self, step: int = 0) -> None:
        pass

    @abstractmethod
    def _update_particles(self, step: int) -> None:
        pass

    # ==================================================================
    # ---  Private methods
    # ==================================================================

    def _plt_well(self, step: int) -> None:
        """ FIXME(tpvasconcelos) not currently working for step potentials"""
        # x, y = self.x[step], self.y[step]
        # for x, y in zip(x, y):
        #     circle_settings = dict(
        #         xy=(x, y), facecolor='none', lw=1,
        #         radius=R_SQUAREWELL * self.sm.RADIUS_PARTICLE)
        #     self.ax.add_patch(Circle(**circle_settings))
        pass

    def _update_slider(self, step: np.float64) -> None:
        # Turn step from 'np.float64' to 'int'
        step = round(float(step))
        self._update_particles(step)
        # update canvas
        self.fig.canvas.draw_idle()
    
    # ==================================================================
    # ---  Public methods || User methods
    # ==================================================================
    
    def plt_slider(self) -> None:
        """ Plot 2D slider. """
        from matplotlib.widgets import Slider

        self._add_particles()
        
        # set up slider
        ax_step_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
        slider = Slider(ax=ax_step_slider, label='Step',
                        valmin=0, valmax=self.sm.STEPS - 1,
                        valinit=0, valfmt='%1.f')
        slider.on_changed(self._update_slider)
        
        self._safe_show(self.plt_slider.__name__)
    
    def anim(self, loop: bool = True) -> None:
        """ Plot 2D animation loop. """
        import matplotlib.animation as animation

        def anim_init() -> Tuple[Artist, ...]:
            self._add_particles()
            return self.artists

        def anim_update(step) -> Tuple[Artist, ...]:
            self._update_particles(step)
            return self.artists

        # Delay between frames in milliseconds (24fps)
        spf = int(1000 * (1 / 24))
        
        # noinspection PyTypeChecker,PyUnusedLocal
        anim = animation.FuncAnimation(
            fig=self.fig,
            func=anim_update,
            frames=np.arange(0, self.sm.STEPS - 1, self.frame_step),
            init_func=anim_init,
            interval=spf,
            blit=True,
            repeat=loop,
            repeat_delay=2 * spf
            )

        self._safe_show(self.anim.__name__)
    
    def export_animation(self,
                         dpi: int = 72,
                         timeit: bool = False
                         ) -> None:
        """ Export animation frames. """
        
        if timeit:
            self.pbarr.set_start()

        self._add_particles()

        for step in range(0, self.sm.STEPS, self.frame_step):
            
            # Remove and redraw/re-plot particles
            self._update_particles(step)
            
            # Save figure
            fname = "{}/img{:06}.png".format(self.sm.png_path, step)
            self.fig.savefig(fname, dpi=dpi, bbox_inches='tight')
            
            self.pbarr.log_progress(step)
        
        plt.clf()
        plt.close('all')
        
        if timeit:
            self.pbarr.set_finish()
            self.pbarr.log_duration()


class MLPAnimationScatter(_BaseMLPAnimation):
    """
    Best for for simulations with large number of particles and/or number of
    steps. It uses ``matplotlib.axes.Axes.scatter`` to plot the particles,
    which results in
      * not being zoom-friendly
      * particle radius **not** being accurately represented
      * faster draw times
    """
    def __init__(self, *args, **kwargs):
        super(MLPAnimationScatter, self).__init__(*args, **kwargs)

        # defined in ``_add_particles()``
        self.ax_scatter: PathCollection = None

    @property
    def artists(self) -> Tuple[PathCollection]:
        return (self.ax_scatter, )

    def _add_particles(self, step: int = 0) -> None:
        # This makes sure that the ``markersize`` is somewhat
        # representative of the actual particle diameter
        # TODO: document the logic used here!
        if self.sm.NDIM == 1:
            scatter_ms = (self.sm.VOL_FRACTION / 0.74) \
                         * (800 / (2 * self.sm.NUM_PARTICLES)) \
                         * (2 * self.sm.RADIUS_PARTICLE)
        else:
            scatter_ms = ((self.sm.VOL_FRACTION / 0.7) ** 0.5) \
                         * (1000 / (2 * self.sm.NUM_PARTICLES ** 0.5)) \
                         * (2 * self.sm.RADIUS_PARTICLE)

        self.ax_scatter: PathCollection = self.ax.scatter(
            x=self.r_coords[0][0],
            y=self.r_coords[0][1],
            color=self.colors[0] if self.colorspeed else self.dflt_clr,
            s=scatter_ms ** 2,
            alpha=0.9,
            lw=0,
        )

    def _update_particles(self, step: int) -> None:
        self.ax_scatter.set_offsets(self.r_vecs[step])
        self.ax_scatter.set_facecolor(self.colors[step] if self.colorspeed else self.dflt_clr)


class MLPAnimationCircles(_BaseMLPAnimation):
    """
    Not ideal for for simulations with large number of particles and/or number
    of  steps. It uses ``matplotlib.patches.Circle`` to plot the particles,
    which results in
      * being zoom-friendly
      * particle radius being accurately represented
      * faster draw times
    """

    @property
    def artists(self) -> Tuple[Circle, ...]:
        return tuple(filter(lambda c: isinstance(c, Circle), self.ax.get_children()))

    def _add_particles(self, step: int = 0) -> None:
        for i in range(self.sm.NUM_PARTICLES):
            self.ax.add_patch(
                Circle(
                    xy=(self.r_coords[step][0][i], self.r_coords[step][1][i]),
                    fc=self.colors[step][i] if self.colorspeed else self.dflt_clr,
                    radius=self.sm.RADIUS_PARTICLE,
                    alpha=0.9,
                    lw=0,
                )
            )

    def _update_particles(self, step: int) -> None:
        for a in self.artists:
            a.remove()
        self._add_particles(step=step)
