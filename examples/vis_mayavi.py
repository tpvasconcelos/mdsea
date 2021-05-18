import mdsea as md
from mdsea.vis.mayavi import MayaviAnimation

sm = md.SysManager.load(simid="_mdsea_docs_example")

anim = MayaviAnimation(sm)
