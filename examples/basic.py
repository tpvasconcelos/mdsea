import mdsea as md
from mdsea.vis.matplotlib import MLPAnimationCircles

# Instantiate system manager  ---
sm = md.SysManager.new(
    simid="_mdsea_docs_example",
    ndim=2,
    num_particles=6 ** 2,
    steps=500,
    vol_fraction=0.2,
    radius_particle=0.5,
    pot=md.Potential.lennardjones(1, 1),
)

# Instantiate simulation  ---
sim = md.ContinuousPotentialSolver(sm)

# Run the simulation  ---
sim.run_simulation()

# Run 2D animation with matplotlib  ---
anim = MLPAnimationCircles(sm, frame_step=6, colorspeed=True)
anim.anim()
