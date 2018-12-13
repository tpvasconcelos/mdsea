#!/usr/local/bin/python3
# coding: utf-8
from tests import mdsea as md

# Calculate the 'number of steps'  ---
STEPS = 3000

# Potential function ( /python-class )
pot = md.Potential.boundedmie(a=1.2, epsilon=1, sigma=1, m=12, n=6)
p_radius = pot.kwargs['sigma'] / 2

# Instantiate system  ---
NDIM = 2
sm = md.SysManager.new(
    # ID (optional)
    simid="_mdsea_testsimulation",
    # Mandatory fields
    ndim=NDIM,
    num_particles=6 ** NDIM,
    vol_fraction=0.4,
    radius_particle=p_radius,
    # Optional fields
    pbc=False,
    temp=.1,
    # mass=1,
    # gravity=True,
    # delta_t=None,
    log_level=10,  # DEBUG LEVEL
    steps=STEPS,
    pot=pot,
    isothermal=True,
    # quench_temps=[],
    # quench_steps=[],
    # quench_timings=[],
    r_cutoff=2.5 * pot.potminimum(),
    # restitution_coeff=0.4,
    # reduced_units=False
    )

# Generate initial velocities  ---
vgen = md.VelGen(nparticles=sm.NUM_PARTICLES, ndim=sm.NDIM)
sm.v_vec = vgen.mb(sm.MASS, sm.TEMP, sm.K_BOLTZMANN)

# Generate initial positions  ---
cgen = md.PosGen(nparticles=sm.NUM_PARTICLES, ndim=sm.NDIM, boxlen=sm.LEN_BOX)
sm.r_vec = cgen.simplecubic()


class Sim(md.ContinuousPotentialSolver):
    
    def run(self):
        self.pbarr.set_start()
        
        simrange = range(self.step, self.sm.STEPS)
        
        # _init_pairs is slow so we only
        # call it if we really need it
        if len(simrange) > 0 and self.pairs is None:
            self._init_pairs()
        
        # Make sure that we update
        # the files at least once
        if len(simrange) == 0:
            self.update_files()
        
        dists = []
        
        # ---  the actual simulation  ---
        for self.step in simrange:
            self.pbarr.log_progress(self.step)
            self.advance()
            self.track_energies()
            self.update_files()
            dists.extend(self.distances)
        # ---  the actual simulation  ---
        
        self.pbarr.set_finish()
        self.pbarr.log_duration()
        
        return dists
        # return self.distances


# Run the simulation  ---
sim = Sim(sm)
ds = sim.run()

# # Run 2D animation with matplotlib  ---
# from mdsea.vis.mpl import Animation
#
# anim = Animation(sm, frame_step=6)
# anim.anim(colorspeed=True)
# exit()

import matplotlib.pyplot as plt

freq, dist, p = plt.hist(ds, bins=200, range=(0, cgen.boxlen / 2),
                         density=True)

dr = dist[1]
rho = 1 / (cgen.boxlen ** 2)

rdf = [0.0]
for i, (f, r) in enumerate(zip(freq, dist)):
    if i == 0:
        continue
    n = rho * 4 * 3.14 * r ** 2 * dr
    rdf.append(f / n)

print(dist[:-1])
print(rdf)

exit()

plt.clf()
plt.plot(dist[:-1], rdf)
plt.show()
