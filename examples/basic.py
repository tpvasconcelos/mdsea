import mdsea as md

# Instantiate system manager  ---
sm = md.SysManager.new(simid="_mdsea_testsimulation",
                       ndim=2, num_particles=6 ** 2, steps=500,
                       vol_fraction=0.2, radius_particle=0.5,
                       pot=md.Potential.lennardjones(1, 1))

# Instantiate simulation  ---
sim = md.ContinuousPotentialSolver(sm)

# Run the simulation  ---
sim.run_simulation()

# Run 2D animation with matplotlib  ---
from mdsea.vis.mpl import MLPAnimationCircles

anim = MLPAnimationCircles(sm, frame_step=6, colorspeed=True)
anim.anim()
