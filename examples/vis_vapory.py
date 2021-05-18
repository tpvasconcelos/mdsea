import mdsea as md
from mdsea.vis.vapory import VaporyAnimation

sm = md.SysManager.load(simid="_mdsea_docs_example")

anim = VaporyAnimation(sm)
anim.render_frame(0, 0)
# anim.render()

# md.make_mp4("{}/mpl.mp4".format(sm.mp4_path), sm.png_path,
#             fps=24, timeit=True)
