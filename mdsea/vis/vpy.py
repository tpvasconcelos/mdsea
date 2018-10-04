#!/usr/local/bin/python
# coding: utf-8
import logging

from PIL.ImageGrab import grab
from vpython import *

from mdsea import loghandler
from mdsea.analytics import SysManager, Vis

log = logging.getLogger(__name__)
log.addHandler(loghandler)


class VpythonAnimation(Vis):
    def __init__(self, sm: SysManager, frame_step: int = 1) -> None:
        super(VpythonAnimation, self).__init__(sm, frame_step)
        
        scene.caption = \
            """
            Right button drag or Ctrl-drag to rotate "camera" to view scene.
            To zoom, drag with middle button or Alt/Option depressed,
            or use scroll wheel. On a two-button mouse, middle is left + right.
            Shift-drag to pan left/right and up/down.
            Touch screen: pinch/extend to zoom, swipe or two-finger rotate.
            """
        
        self.particles = []
        
        self.scene_box = (21,
                          215,
                          2.5 * scene.width - 20,
                          2.5 * scene.height + 215)
        
        self.initialize()
    
    def initialize(self):
        
        for i in range(self.sm.NUM_PARTICLES):
            clr = vector(*self.color(self.speeds[0][i], alpha=False))
            p = sphere(pos=vector(self.y[0][i], self.z[0][i], self.x[0][i]),
                       color=clr, radius=self.sm.RADIUS_PARTICLE)
            self.particles.append(p)
    
    def render_frame(self, step):
        for i in range(self.sm.NUM_PARTICLES):
            # Update position
            self.particles[i].pos = vector(self.y[step][i],
                                           self.z[step][i],
                                           self.x[step][i])
            # Update color
            self.particles[i].color = \
                vector(*self.color(self.speeds[step][i], alpha=False))
    
    def run(self, export: bool = False):
        if export:
            input("[!] Position scene at top left corner of your screen. "
                  "Once you're done, hit 'Enter'.")
            
            time.sleep(1)
        n = 1
        for step in range(self.sm.STEPS):
            if not (step % self.frame_step):
                if export:
                    time.sleep(0.5)
                    grab(self.scene_box).save(
                        "{}/img{:06}.png".format(self.sm.png_path, n))
                self.render_frame(step)
                n += 1
