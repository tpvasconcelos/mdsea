#!/usr/local/bin/python
# coding: utf-8
from mdsea.vis.vpy import VpythonAnimation
from tests import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")

anim = VpythonAnimation(sm)
anim.run()

if True:
    import time
    
    while 1:
        anim.run()
        time.sleep(1)
