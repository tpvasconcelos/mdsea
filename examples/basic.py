import mdsea as md
from mdsea.helpers import setup_logging
from mdsea.vis.matplotlib import MLPAnimationScatter

setup_logging(level="DEBUG")

# Instantiate system manager  ---
sm = md.SysManager.new(
    simid="_mdsea_docs_example",
    ndim=2,
    num_particles=6 ** 2,
    steps=400,
    vol_fraction=0.2,
    radius_particle=0.5,
)

# Instantiate simulation  ---
sim = md.ContinuousPotentialSolver(sm, pot=md.Potential.lennardjones(1, 1))

# Run the simulation  ---
sim.run_simulation()

# Run 2D animation with matplotlib  ---
anim = MLPAnimationScatter(sm, frame_step=6, colorspeed=True)
anim.anim()
