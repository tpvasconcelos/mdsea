from mdsea.vis.vpy import VpythonAnimation
import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")

anim = VpythonAnimation(sm)
anim.run()
