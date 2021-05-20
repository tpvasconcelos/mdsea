from mdsea.core import SysManager
from mdsea.helpers import setup_logging
from mdsea.potentials import Potential
from mdsea.simulator import ContinuousPotentialSolver
from mdsea.vis.matplotlib import MLPAnimationScatter

setup_logging(level="DEBUG")

# Instantiate system manager  ---
sm = SysManager.new(
    simid="_mdsea_docs_example",
    ndim=2,
    num_particles=6 ** 2,
    steps=500,
    vol_fraction=0.2,
    radius_particle=0.5,
)

# Instantiate simulation  ---
sim = ContinuousPotentialSolver(sm, pot=Potential.lennardjones(1, 1))

# Run the simulation  ---
sim.run_simulation()

# Run 2D animation with matplotlib  ---
anim = MLPAnimationScatter(sm, frame_step=6, colorspeed=True)
anim.anim()
