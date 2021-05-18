from mdsea.vis.mpl import MLPAnimationCircles
import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")

anim = MLPAnimationCircles(sm, frame_step=5, colorspeed=True)

# anim.plt_slider()
# anim.anim()

anim.export_animation(dpi=36, timeit=True)
md.make_mp4(f"{sm.mp4_path}/mpl.mp4", sm.png_path, fps=24, timeit=True)
