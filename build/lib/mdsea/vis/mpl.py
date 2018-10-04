#!/usr/local/bin/python
# coding: utf-8
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mdsea.analytics import Vis
from mdsea.core import SysManager
from mdsea.helpers import ProgressBar
from typing import Tuple
import numpy as np
from typing import Optional
import logging
from mdsea import loghandler
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import Colormap

log = logging.getLogger(__name__)
log.addHandler(loghandler)


def color(cmap: Colormap, speed: float, speed_limit: float,
          alpha: bool = True) -> tuple:
    # TODO: vectorize this such that 'speed' can be an array
    # speed_ratio = round(cmap.N * speed / speed_limit)
    speed_ratio = cmap.N - int(cmap.N * speed / speed_limit)
    if cmap.name != 'autumn':
        # Reverse the colors for the 'autumn' color map.
        speed_ratio = cmap.N - speed_ratio
    if not alpha:
        # Remove alpha (transparency)
        return cmap(speed_ratio)[:-1]
    return cmap(speed_ratio)


class MPL(Vis):
    def __init__(self,
                 sm: SysManager,
                 frame_step: int = 1
                 ) -> None:
        super(MPL, self).__init__(sm, frame_step)
        
        if self.sm.NDIM == 1:
            zeroes_ = (self.sm.LEN_BOX / 2) + np.zeros((self.sm.STEPS,
                                                        self.sm.NUM_PARTICLES))
            self.r_coords = np.stack([self.r_coords[:, 0], zeroes_], axis=1)
            self.r_vecs = np.stack([self.r_coords[:, 0], zeroes_], axis=-1)
        
        # TODO: @property doesn't work!!
        self._dark_theme = False
    
    @property
    def dark_theme(self):
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
        plt.plot(self.mean_pot_energies, lw=lw, label=label_pe)
        plt.plot(self.mean_kin_energies, lw=lw, label=label_ke)
        plt.plot(self.total_energy, lw=lw, label=label_te)
        plt.legend(loc='best', prop={'size': font_size})
        plt.grid()
        self._safe_show()
    
    def plt_temp(self,
                 lw: float = 2,
                 fontsize: float = 18,
                 label: str = r'$T^*$') -> None:
        plt.plot(self.temp, lw=lw, label=label)
        plt.legend(loc='best', prop={'size': fontsize})
        plt.grid()
        self._safe_show()
    
    def plt_sd(self,
               fontsize: float = 18,
               label: str = r'$Speed Distribution$') -> None:
        """ Speed distribution plot. """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # generate histogram of velocities
        ax.hist(self.speeds, bins=100, normed=True, label=label)
        # compare this histogram to f(speeds_range_monte_carlo)
        # this is MB_speed that we wrote earlier
        ax.set_xlabel(r'Speed $(m/s)$', size=fontsize)
        ax.set_ylabel('PDF', size=fontsize)
        plt.legend(loc='best', prop={'size': fontsize})
        plt.grid()
        self._safe_show()
    
    def plt_rdf(self,
                lw: float = 2,
                fontsize: float = 18,
                label: str = r'$g(r)$') -> None:
        """ Radial Distribution Function (RDF) plot. """
        plt.plot(self.temp, lw=lw, label=label)
        plt.legend(loc='best', prop={'size': fontsize})
        plt.grid()
        self._safe_show()
    
    @staticmethod
    def _safe_show() -> None:
        try:
            plt.show()
        finally:
            plt.close('all')


class Animation(MPL):
    def __init__(self,
                 # MPL kwargs:
                 sm: SysManager,
                 frame_step: int = 1,
                 # Animation kwargs:
                 scatter: bool = False,
                 color: str = 'orange',
                 figsize: Tuple[int, int] = (15, 10)
                 ) -> None:
        """
        
        :param scatter
            If True   - matplotlib.axes.Axes.scatter
                      - [+] Fast draw time (ideal for simulations with large
                            number of particles and/or number of steps)
                      - [-] particle radius NOT accuratly represented
                      - [-] NOT zoom-friendly
            If False  - matplotlib.patches.Circle
                      - [-] Slow draw time (NON-ideal for simulations with
                            large number of particles and/or number of steps)
                      - [+] Particle radius accuratly represented
                      - [+] Zoom-friendly
        
        :param draw_wells
            Draw the potential wells (only for square well)
        
        """
        
        # ==============================================================
        # ---  super
        # ==============================================================
        
        super(Animation, self).__init__(sm, frame_step)
        
        # ==============================================================
        # ---  Parce kwarguments
        # ==============================================================
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        self.dflt_clr = color
        self.draw_wells = None
        self.scatter = scatter
        
        # ==============================================================
        # ---  Axes.scatter settings
        # ==============================================================
        
        self.ax_scatter = None
        # This makes sure that the markersize is somewhat
        # representative of the actual particle diameter
        if self.sm.NDIM == 1:
            scatter_ms = (self.sm.VOL_FRACTION / 0.74) \
                         * (800 / (2 * self.sm.NUM_PARTICLES)) \
                         * (2 * self.sm.RADIUS_PARTICLE)
        else:
            scatter_ms = ((self.sm.VOL_FRACTION / 0.7) ** 0.5) \
                         * (1000 / (2 * self.sm.NUM_PARTICLES ** 0.5)) \
                         * (2 * self.sm.RADIUS_PARTICLE)
        
        self.scatter_size = scatter_ms ** 2
        
        # ==============================================================
        # ---  Others
        # ==============================================================
        
        self.colorspeed = False
        
        self.pbarr = ProgressBar("Saving animation frame",
                                 self.sm.STEPS, __name__)
    
    # ==================================================================
    # ---  Private methods
    # ==================================================================
    
    def _plt_box(self) -> None:
        """ Plot the box where the particles are contained in. """
        w = self.sm.LEN_BOX
        if self.sm.NDIM == 1:
            h = 2 * self.sm.RADIUS_PARTICLE
            origin = (0, self.sm.LEN_BOX / 2 - h / 2)
        else:
            h = self.sm.LEN_BOX
            origin = (0, 0)
        
        lw = 1
        inner_box = dict(xy=origin, width=w, height=h, lw=lw, fill=False)
        self.ax.add_patch(patches.Rectangle(**inner_box))
    
    def _scatter_init(self):
        kwargs = dict(s=self.scatter_size, lw=0, alpha=0.9)
        if self.colorspeed:
            kwargs['color'] = self.colors[0]
        self.ax_scatter = self.ax.scatter(self.r_coords[0][0],
                                          self.r_coords[0][1], **kwargs)
        return self.ax_scatter,
    
    def _plt_particles_scatter(self, step: int) -> None:
        clr = self.dflt_clr
        if self.colorspeed:
            clr = self.colors[step]
        self.ax_scatter.set_offsets(self.r_vecs[step])
        self.ax_scatter.set_facecolor(clr)
        # print(self.r_vecs[step])
        # print(clr)
        # exit()
    
    def _plt_particles_circles(self, step: int) -> None:
        clr = self.dflt_clr
        
        for i in range(self.sm.NUM_PARTICLES):
            if self.colorspeed:
                clr = self.colors[step][i]
            circle_settings = dict(xy=(self.r_coords[step][0][i],
                                       self.r_coords[step][1][i]),
                                   radius=self.sm.RADIUS_PARTICLE, lw=0,
                                   fc=clr, alpha=0.9)
            self.ax.add_patch(patches.Circle(**circle_settings))
    
    def _plt_particles(self, step: int) -> None:
        """ Plot particles
        
        :param step
            Simulation step (int)
    
        """
        if self.scatter:
            self._plt_particles_scatter(step)
            return
        # DEFAULT (if not scatter)
        self._plt_particles_circles(step)
    
    def _plt_well(self, step: int) -> None:
        """
        # TODO: R_SQUAREWELL
        x, y = self.x[step], self.y[step]
        for x, y in zip(x, y):
            circle_settings = dict(xy=(x, y), facecolor='none', lw=1,
                                   radius=R_SQUAREWELL * self.sm.RADIUS_PARTICLE)
            self.ax.add_patch(patches.Circle(**circle_settings))
        """
        pass
    
    def _rm_particles(self) -> None:
        if self.scatter and self.ax_scatter:
            self.ax_scatter.remove()
            return
        
        # DEFAULT (if not scatter)
        circles = [c for c in self.fig.axes[0].get_children()
                   if isinstance(c, patches.Circle)]
        for circle in circles:
            circle.remove()
    
    def _set_preferences(self) -> None:
        min_ = -0.1
        max_ = self.sm.LEN_BOX + 0.1
        self.ax.axis(xmin=min_, xmax=max_, ymin=min_, ymax=max_)
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()  # remove axis0
    
    def _update_slider(self, step: np.float64) -> None:
        self._rm_particles()
        # Turn step from 'np.float64' to 'int'
        step = round(float(step))
        self._plt_particles(step)
        if self.draw_wells:
            self._plt_well(step)
        self.fig.canvas.draw_idle()  # update canvas
    
    def _update_animloop(self, step):
        self._plt_particles_scatter(step)
        return self.ax_scatter,
    
    def _plt_init(self) -> None:
        # set up initial conditions
        self._plt_box()
        self._set_preferences()
        if self.colorspeed:
            self._clrs_init()
    
    def _clrs_init(self, cmap: Optional[Colormap] = None):
        if cmap is None:
            # DIVERGING: ['coolwarm', 'RdBu_r', 'jet']
            # SEQUENTIAL: ['gist_heat', 'autumn', 'hot']
            num_colors = 256 / 2
            cmap = cm.get_cmap(name='autumn', lut=num_colors)
        
        self.colors = [[color(cmap, s, self.maxspeed) for s in ss]
                       for ss in self.speeds]
    
    # ==================================================================
    # ---  Public methods || User methods
    # ==================================================================
    
    def plot_slider(self,
                    scatter: Optional[bool] = None,
                    draw_wells: bool = False,
                    colorspeed: bool = False
                    ) -> None:
        """ Plot 2D slider. """
        if isinstance(scatter, bool):
            self.scatter = scatter
        self.draw_wells = draw_wells
        self.colorspeed = colorspeed
        
        self._plt_init()
        self._plt_particles(step=0)
        if self.draw_wells and not self.scatter:
            self._plt_well(step=0)
        
        # set up slider
        ax_step_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
        slider = Slider(ax=ax_step_slider, label='Step',
                        valmin=0, valmax=self.sm.STEPS - 1,
                        valinit=0, valfmt='%1.f')
        slider.on_changed(self._update_slider)
        
        log.info("Plotting plot_slider...")
        self._safe_show()
    
    def anim(self,
             scatter: Optional[bool] = None,
             draw_wells: bool = False,
             colorspeed: bool = False,
             loop: bool = True
             ) -> None:
        """ Plot 2D animation loop. """
        if isinstance(scatter, bool):
            self.scatter = scatter
        self.draw_wells = draw_wells
        self.colorspeed = colorspeed
        
        self._plt_init()
        
        # Delay between frames in milliseconds (24fps)
        spf = int(1000 * (1 / 24))
        
        # noinspection PyTypeChecker,PyUnusedLocal
        anim = animation.FuncAnimation(
            fig=self.fig, func=self._update_animloop,
            frames=np.arange(0, self.sm.STEPS - 1, self.frame_step),
            interval=spf, blit=True, init_func=self._scatter_init,
            repeat=loop, repeat_delay=2 * spf
            )
        
        log.info("Plotting anim_loop...")
        self._safe_show()
    
    def export_animation(self,
                         dpi: int = 72,
                         timeit: bool = False
                         ) -> None:
        """ Export animation frames. """
        
        if timeit:
            self.pbarr.set_start()
        
        # Set up initial conditions
        self._plt_box()
        self._set_preferences()
        
        for step in range(0, self.sm.STEPS, self.frame_step):
            
            # Remove and redraw/replot particles
            self._rm_particles()
            self._plt_particles(step)
            
            # Save figure
            fname = "{}/img{:06}.png".format(self.sm.png_path, step)
            self.fig.savefig(fname, dpi=dpi, bbox_inches='tight')
            
            self.pbarr.log_progress(step)
        
        plt.clf()
        plt.close('all')
        
        if timeit:
            self.pbarr.set_finish()
            self.pbarr.log_duration()
