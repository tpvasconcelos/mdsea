#!/usr/local/bin/python
# coding: utf-8
import logging

from vapory import *

from mdsea import loghandler
from mdsea.analytics import SysManager, Vis
from mdsea.helpers import rgb2unit

log = logging.getLogger(__name__)
log.addHandler(loghandler)


class VaporyAnimation(Vis):
    def __init__(self, sm: SysManager, frame_step: int = 1) -> None:
        super(VaporyAnimation, self).__init__(sm, frame_step)
        
        # vapory objects
        self.scene: Scene = None
        self.lights: list = None
        self.objects: list = None
        self.camera: Camera = None
        self.radiosity: Radiosity = None
        self.background: Background = None
        
        # Colors
        self.clr_sun = (1, 1, 1)  # (1, 0.8, 0.7)
        self.clr_background = (0.1, 0.1, 0.12)  # rgb2unit((70, 80, 100))
        self.clr_particle = rgb2unit((50, 75, 175))  # (1, 0.75, 0)
        
        # INIT
        self.initialize()
    
    def initialize(self):
        # Set up camera
        camera_location = [2 * self.sm.LEN_BOX,
                           1.25 * self.sm.LEN_BOX,
                           -0.5 * self.sm.LEN_BOX]
        
        self.camera = Camera(
            'location', camera_location,
            'look_at', [self.sm.LEN_BOX / 2,
                        self.sm.LEN_BOX / 2.5,
                        self.sm.LEN_BOX / 2])
        
        # Set up background
        self.background = Background("color", self.clr_background)
        
        # Set up lighting
        self.lights = [
            LightSource(camera_location,
                        'color', [1, 1, 1]),
            LightSource([100 * self.sm.LEN_BOX,
                         100 * self.sm.LEN_BOX,
                         - 100 * self.sm.LEN_BOX],
                        'color', self.clr_sun)
            ]
        
        # Global Settings
        # self.radiosity = Radiosity()
        
        # Goup Objects
        self.objects = [self.background, *self.lights]
    
    def render_frame(self, step, outn):
        
        particles = []
        for i in range(self.sm.NUM_PARTICLES):
            clr = self.color(self.speeds[0][i], alpha=False)
            p = Sphere([self.y[step][i],
                        self.z[step][i],
                        self.x[step][i]],
                       self.sm.RADIUS_PARTICLE,
                       Texture(Pigment('color', clr)))
            particles.append(p)
        
        self.scene = Scene(self.camera, objects=self.objects + particles)
        
        self.scene.render("{}/img{:06}.png".format(self.sm.png_path, outn),
                          width=600, height=400)
    
    def render(self):
        n = 1
        for step in range(self.sm.STEPS):
            if not (step % self.frame_step):
                self.render_frame(step, n)
                n += 1
