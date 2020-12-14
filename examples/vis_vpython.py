#!/usr/local/bin/python
# coding: utf-8
from mdsea.vis.vpy import VpythonAnimation
import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")

anim = VpythonAnimation(sm)
anim.run()
