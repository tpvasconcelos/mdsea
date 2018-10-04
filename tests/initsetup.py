#!/usr/local/bin/python3
# coding: utf-8
from tests import mdsea as md

# Calculate the 'number of steps'
SECONDS = 1
FRAME_STEP = 5
FRAMES_PER_SECOND = 24
STEPS = int(SECONDS * FRAMES_PER_SECOND * FRAME_STEP)

# Potential function (...and python class)
# pot = md.Potential.boundedmie(epsilon=1, sigma=1, m=12, n=6, a=0.1)
pot = md.Potential.boundedmie(a=0.2, epsilon=1, sigma=1, m=12, n=6)
p_radius = pot.kwargs['sigma'] / 2

# Instantiate system
NDIM = 3
sm = md.SysManager.new(
    # ID (optional)
    simid="_mdsea_testsimulation",
    # Mandatory fields
    ndim=NDIM,
    num_particles=2 ** NDIM,
    vol_fraction=0.1,
    radius_particle=p_radius,
    # Optional fields
    pbc=False,
    temp=1,
    # mass=1,
    # gravity=True,
    # delta_t=None,
    log_level=10,  # DEBUG LEVEL
    steps=STEPS,
    pot=pot,
    # isothermal=True,
    # quench_temps=[],
    # quench_steps=[],
    # quench_timings=[],
    r_cutoff=2.5 * pot.potminimum(),
    # restitution_coeff=0.4,
    # reduced_units=False
    )

# Generate initial velocities
vgen = md.VelGen(nparticles=sm.NUM_PARTICLES,
                 ndim=sm.NDIM)
sm.v_vec = vgen.mb(sm.MASS, sm.TEMP, sm.K_BOLTZMANN)

# Generate initial positions
cgen = md.CoordGen(nparticles=sm.NUM_PARTICLES,
                   ndim=sm.NDIM,
                   boxlen=sm.LEN_BOX)
sm.r_vec = cgen.simplecubic()

# ======================================================================
# ---  Run the simulation
# ======================================================================

sim = md.ContinuousPotentialSolver(sm, algorithm='verlet')
sim.run_simulation()
