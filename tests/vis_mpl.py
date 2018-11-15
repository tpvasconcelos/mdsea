#!/usr/local/bin/python3
# coding: utf-8
from mdsea.vis.mpl import Animation
from tests import mdsea as md

FRAME_STEP = 5

sm = md.SysManager.load(simid="_mdsea_testsimulation")

anim = Animation(sm, frame_step=FRAME_STEP)

# anim.dark_theme = True

# anim.plot_slider(scatter=True, colorspeed=True)
anim.anim(scatter=False, colorspeed=True)

# anim.export_animation(dpi=36, timeit=True)
# md.make_mp4("{}/mpl.mp4".format(sm.mp4_path), sm.png_path,
#             fps=24, timeit=True)
