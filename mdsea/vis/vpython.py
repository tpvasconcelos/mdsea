import logging
import time

from mdsea.analytics import SysManager, Vis
from mdsea.vis.matplotlib import speed2color
from PIL.ImageGrab import grab
from vpython import scene, sphere, vector

log = logging.getLogger(__name__)


class VpythonAnimation(Vis):
    def __init__(self, sm: SysManager, frame_step: int = 1) -> None:
        super().__init__(sm, frame_step)

        scene.caption = """
            Right button drag or Ctrl-drag to rotate "camera" to view scene.
            To zoom, drag with middle button or Alt/Option depressed,
            or use scroll wheel. On a two-button mouse, middle is left + right.
            Shift-drag to pan left/right and up/down.
            Touch screen: pinch/extend to zoom, swipe or two-finger rotate.
            """

        self.particles = []

        self.scene_box = (21, 215, 2.5 * scene.width - 20, 2.5 * scene.height + 215)

        self.initialize()

    def initialize(self):

        for i in range(self.sm.NUM_PARTICLES):
            clr = vector(*speed2color(speed=self.speeds[0][i], speed_limit=self.maxspeed, alpha=False))
            p = sphere(
                pos=vector(self.r_vecs[0][i][1], self.r_vecs[0][i][2], self.r_vecs[0][i][0]),
                color=clr,
                radius=self.sm.RADIUS_PARTICLE,
            )
            self.particles.append(p)

    def render_frame(self, step):
        for i in range(self.sm.NUM_PARTICLES):
            # Update position
            self.particles[i].pos = vector(self.r_vecs[step][i][1], self.r_vecs[step][i][2], self.r_vecs[step][i][0])
            # Update color
            self.particles[i].color = vector(
                *speed2color(speed=self.speeds[step][i], speed_limit=self.maxspeed, alpha=False)
            )

    def run(self, export: bool = False):
        if export:
            input("[!] Position scene at top left corner of your screen. " "Once you're done, hit 'Enter'.")

            time.sleep(1)
        n = 1
        for step in range(self.sm.STEPS):
            if not (step % self.frame_step):
                if export:
                    time.sleep(0.5)
                    grab(self.scene_box).save(f"{self.sm.png_path}/img{n:06}.png")
                time.sleep(1 / 24.0)
                self.render_frame(step)
                n += 1
