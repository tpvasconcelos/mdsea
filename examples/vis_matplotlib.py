import mdsea as md
from mdsea.helpers import setup_logging
from mdsea.vis.matplotlib import MLPAnimationCircles

setup_logging(level="DEBUG")

sm = md.SysManager.load(simid="_mdsea_docs_example")

anim = MLPAnimationCircles(sm, frame_step=5, colorspeed=True)

# anim.plt_slider()
anim.anim()

# anim.export_animation(dpi=36, timeit=True)
# md.make_mp4(f"{sm.mp4_path}/mpl.mp4", sm.png_path, fps=24, timeit=True)
