import mdsea as md
from mdsea.vis.vpython import VpythonAnimation

sm = md.SysManager.load(simid="_mdsea_docs_example")

anim = VpythonAnimation(sm)
anim.run()
