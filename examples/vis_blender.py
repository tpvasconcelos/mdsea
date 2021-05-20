from mdsea.core import SysManager
from mdsea.helpers import setup_logging
from mdsea.vis.blender import BlenderAnimation

setup_logging(level="DEBUG")

FRAME_STEP = 5


def main():
    sm = SysManager.load(simid="_mdsea_docs_example")

    anim = BlenderAnimation(sm, frame_step=FRAME_STEP)

    # Setup
    anim.quick_setup(engine="CYCLES")
    anim.set_view(shade="SOLID")

    # Extra objects
    anim.add_light("front")
    anim.add_light("left")
    anim.add_light("above")
    anim.add_light("diagonal")
    anim.add_floor()
    # anim.add_glasswalls()

    # Create particles last for performance!
    anim.create_particle_system()

    anim.run()

    # anim.render()
    # md.make_mp4("{}/mpl.mp4".format(sm.mp4_path), sm.png_path,
    #             fps=24, timeit=True)


if __name__ == "__main__":
    main()
