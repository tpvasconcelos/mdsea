#!/usr/local/bin/python
# coding: utf-8
from tests import mdsea as md
from mdsea.vis.vpr import VaporyAnimation

sm = md.SysManager.load(simid="_mdsea_testsimulation")

anim = VaporyAnimation(sm)
anim.render_frame(0, 0)
# anim.render()

# md.make_mp4("{}/mpl.mp4".format(sm.mp4_path), sm.png_path,
#             fps=24, timeit=True)
