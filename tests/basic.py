#!/usr/local/bin/python3
# coding: utf-8
import mdsea as md

# Instantiate system manager
sm = md.SysManager.new(ndim=2, num_particles=8 ** 2, steps=1000,
                       vol_fraction=0.4, radius_particle=0.5,
                       pot=md.Potential.lennardjones(1, 1))

# Generate initial velocities
vgen = md.VelGen(ndim=sm.NDIM, nparticles=sm.NUM_PARTICLES)
sm.v_vec = vgen.mb(sm.MASS, sm.TEMP, sm.K_BOLTZMANN)

# Generate initial positions
cgen = md.CoordGen(ndim=sm.NDIM, nparticles=sm.NUM_PARTICLES,
                   boxlen=sm.LEN_BOX)
sm.r_vec = cgen.simplecubic()

# Run the simulation
sim = md.ContinuousPotentialSolver(sm)
sim.run_simulation()

# Run 2D animation with matplotlib
from mdsea.vis.mpl import Animation

anim = Animation(sm, frame_step=6)
anim.anim()
