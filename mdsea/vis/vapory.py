import logging

from mdsea.analytics import SysManager, Vis
from mdsea.helpers import rgb2unit
from mdsea.vis.matplotlib import speed2color
from vapory import (
    Background,
    Camera,
    LightSource,
    Pigment,
    Radiosity,
    Scene,
    Sphere,
    Texture,
)

log = logging.getLogger(__name__)


class VaporyAnimation(Vis):
    def __init__(self, sm: SysManager, frame_step: int = 1) -> None:
        super().__init__(sm, frame_step)

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
        camera_location = [2 * self.sm.LEN_BOX, 1.25 * self.sm.LEN_BOX, -0.5 * self.sm.LEN_BOX]

        self.camera = Camera(
            "location", camera_location, "look_at", [self.sm.LEN_BOX / 2, self.sm.LEN_BOX / 2.5, self.sm.LEN_BOX / 2]
        )

        # Set up background
        self.background = Background("color", self.clr_background)

        # Set up lighting
        self.lights = [
            LightSource(camera_location, "color", [1, 1, 1]),
            LightSource([100 * self.sm.LEN_BOX, 100 * self.sm.LEN_BOX, -100 * self.sm.LEN_BOX], "color", self.clr_sun),
        ]

        # Global Settings
        # self.radiosity = Radiosity()

        # Goup Objects
        self.objects = [self.background, *self.lights]

    def render_frame(self, step, outn):

        particles = []
        for i in range(self.sm.NUM_PARTICLES):
            clr = speed2color(speed=self.speeds[step][i], speed_limit=self.maxspeed, alpha=False)
            p = Sphere(
                [self.r_vecs[step][i][1], self.r_vecs[step][i][2], self.r_vecs[step][i][0]],
                self.sm.RADIUS_PARTICLE,
                Texture(Pigment("color", clr)),
            )
            particles.append(p)

        self.scene = Scene(self.camera, objects=self.objects + particles)

        self.scene.render(f"{self.sm.png_path}/img{outn:06}.png", width=600, height=400)

    def render(self):
        n = 1
        for step in range(self.sm.STEPS):
            if not (step % self.frame_step):
                self.render_frame(step, n)
                n += 1
