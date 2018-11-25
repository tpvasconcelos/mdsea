#!/usr/local/bin/python
# coding: utf-8
import os

DIR_SIMFILES = f'{os.getcwd()}/simfiles'

# Main directory (the simulation directory) - LEVEL-0
# sim_path = '{}/{}'.format(DIR_SIMFILES, SIM_ID)
SIM_PATH = f'{DIR_SIMFILES}/' + '{}'

# Directory names
# LEVEL-1
VIS_DNAME = 'vis'
# LEVEL-2
PNG_DNAME = 'png'
MP4_DNAME = 'mp4'
BLENDER_DNAME = 'blender'

# Path to directories
# LEVEL-1
VIS_PATH = f'{SIM_PATH}/{VIS_DNAME}'
# LEVEL-2
PNG_PATH = f'{SIM_PATH}/{VIS_DNAME}/{PNG_DNAME}'
MP4_PATH = f'{SIM_PATH}/{VIS_DNAME}/{MP4_DNAME}'
BLENDER_PATH = f'{SIM_PATH}/{VIS_DNAME}/{BLENDER_DNAME}'

# Main data file (HDF5)
DATA_FNAME = 'data.h5'
DATA_PATH = f'{SIM_PATH}/{DATA_FNAME}'

# Settings file
SETTINGS_FNAME = 'settings.pkl'
SETTINGS_PATH = f'{SIM_PATH}/{SETTINGS_FNAME}'

# Datasets  ---

# Datasets with shape=(STEPS, NDIM, NUM_PARTICLES)
RCOORDS_DSNAME = 'r_coords'
VCOORDS_DSNAME = 'v_coords'

# Datasets with shape=(STEPS,)
POTENERGY_DSNAME = 'pot_energy'
KINENERGY_DSNAME = 'kin_energy'
TEMP_DSNAME = 'temp'
