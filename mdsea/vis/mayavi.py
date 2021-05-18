import logging

from mayavi import mlab
from mdsea.analytics import SysManager, Vis
from mdsea.vis.matplotlib import speed2color

log = logging.getLogger(__name__)


class MayaviAnimation(Vis):
    def __init__(self, sm: SysManager, frame_step: int = 1) -> None:
        super().__init__(sm, frame_step)

        # Disable the rendering, to get bring up the figure quicker:
        figure = mlab.gcf()
        mlab.clf()
        figure.scene.disable_render = True

        for i in range(self.sm.NUM_PARTICLES):
            mlab.points3d(
                self.r_vecs[0][i][0],
                self.r_vecs[0][i][1],
                self.r_vecs[0][i][1],
                color=speed2color(speed=self.speeds[0][i], speed_limit=self.maxspeed, alpha=False),
                resolution=8 * 3,
            )

        # Every object has been created, we can reenable the rendering.
        figure.scene.disable_render = False

        mlab.show()
