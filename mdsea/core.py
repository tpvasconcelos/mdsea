#!/usr/local/bin/python
# coding: utf-8
import logging
import os
import pickle
import shutil
from typing import Optional, Union

import h5py
import numpy as np
from scipy.constants import codata

from mdsea import loghandler
from mdsea.constants import DTYPE, fileman as cfm
from mdsea.helpers import nsphere_volume
from mdsea.potentials import Potential

log = logging.getLogger(__name__)
log.addHandler(loghandler)

DIR_SIMFILES = f"{os.getcwd()}/simfiles"


def _gen_newid() -> str:
    from time import time
    return str(time()).replace('.', '')


class SysManager:
    def __init__(self,
                 # SysManager.load() and SysManager.new()
                 # take care of these two arguments:
                 new_sim: bool,
                 simid: Optional[str],
                 # User mandatory arguments:
                 ndim: int,
                 num_particles: int,
                 vol_fraction: float,
                 radius_particle: float,
                 # Optional arguments:
                 temp: float = 1,
                 mass: float = 1,
                 pbc: bool = True,
                 steps: int = 1,
                 gravity: bool = False,
                 isothermal: bool = False,
                 reduced_units: bool = True,
                 quench_temps: list = list(),
                 quench_steps: list = list(),
                 quench_timings: list = list(),
                 restitution_coeff: float = 1.,
                 delta_t: Optional[float] = None,
                 log_level: Optional[int] = None,
                 r_cutoff: Optional[float] = None,
                 pot: Potential = Potential.ideal(),
                 ) -> None:
        """
        TODO: doc string
        
        The file manager, manages:

            - the main "simulations-folder"
            - each individual simulation directory
            - the directory tree of each simulation directory
            - the simulation files and datasets (including HDF5)

        """
        
        # ==============================================================
        # ---  Parce user arguments/settings
        # ==============================================================
        
        # Settings dictionary (we'll save this to a file)
        self._settings = dict()
        
        # Logging stuff  ---
        
        if log_level is not None:
            logging.basicConfig(level=log_level)
            self._settings.update(log_level=log_level)
        
        # Bunch 'o variables  ---
        
        self.RESTITUTION_COEFF = restitution_coeff
        self.RADIUS_PARTICLE = radius_particle
        self.NUM_PARTICLES = num_particles
        self.VOL_FRACTION = vol_fraction
        self.ISOTHERMAL = isothermal
        self.R_CUTOFF = r_cutoff
        self.DELTA_T = delta_t
        self.GRAVITY = gravity
        self.STEPS = steps
        self.NDIM = ndim
        self.MASS = mass
        self.TEMP = temp
        self.POT = pot
        self.PBC = pbc
        
        self._settings.update(restitution_coeff=restitution_coeff)
        self._settings.update(radius_particle=radius_particle)
        self._settings.update(num_particles=num_particles)
        self._settings.update(vol_fraction=vol_fraction)
        self._settings.update(isothermal=isothermal)
        self._settings.update(r_cutoff=r_cutoff)
        self._settings.update(delta_t=delta_t)
        self._settings.update(gravity=gravity)
        self._settings.update(steps=steps)
        self._settings.update(ndim=ndim)
        self._settings.update(mass=mass)
        self._settings.update(temp=temp)
        self._settings.update(pot=pot)
        self._settings.update(pbc=pbc)
        
        # Quenching stuff  ---
        
        self.QUENCH_T = quench_temps
        self.QUENCH_TIMING = quench_timings
        self.QUENCH_STEP = [round(self.STEPS * t)
                            for t in self.QUENCH_TIMING]
        
        self._settings.update(quench_temps=quench_temps)
        self._settings.update(quench_steps=quench_steps)
        self._settings.update(quench_timings=quench_timings)
        
        # Physical constants  ---
        
        self.GRAVITY_ACCELERATION = codata.value(
            'standard acceleration of gravity')
        
        self.K_BOLTZMANN = 1
        if not reduced_units:
            self.K_BOLTZMANN = codata.value('Boltzmann constant')
        
        self._settings.update(reduced_units=reduced_units)
        
        # Simulation-ID, and stuff  ---
        
        # Start by assuming this is NOT a new simulation,
        # and that we have been given a simulation-ID.
        self.SIM_ID = simid
        self.new_sim = new_sim
        if self.SIM_ID is None:
            # If we have not been given a simulation-ID,
            # we now treat this as a new simulation,
            # and generate a new simulation-ID.
            self.new_sim = True
            self.SIM_ID = _gen_newid()
        
        self._settings.update(new_sim=new_sim)
        self._settings.update(simid=simid)
        
        # ==============================================================
        # ---  Others settings
        # ==============================================================
        
        # TODO: These shouldnt be here? They should be in the simulator?
        self.r_vec = np.zeros((self.NDIM, self.NUM_PARTICLES),
                              dtype=DTYPE)
        self.v_vec = np.zeros((self.NDIM, self.NUM_PARTICLES),
                              dtype=DTYPE)
        
        # Calculation of the 'box lenght' ('self.LEN_BOX')
        particle_volume = nsphere_volume(self.NDIM, self.RADIUS_PARTICLE)
        box_volume = self.NUM_PARTICLES * particle_volume / self.VOL_FRACTION
        self.LEN_BOX = box_volume ** (1 / self.NDIM)
        
        # ==============================================================
        # ---  File Manager stuff
        # ==============================================================
        
        # ==============================================================
        #
        # The directories are arranged in different levels:
        # LEVEL-0   is the main directory (the simulation directory)
        # LEVEL-N   directories are immediately inside LEVEL-(N-1)
        #           directories
        #
        # Example:
        #
        # LEVEL-0/
        # ├── LEVEL-1/
        # |   ├── LEVEL-2/
        # |   └── LEVEL-2/
        # ├── LEVEL-1/
        # |   ├── LEVEL-2/
        # |   |   ├── foo.py
        # |   |   └── bar.py
        # └   └── LEVEL-2/
        #         ├── LEVEL-3/
        #         └── LEVEL-3/
        #
        # ==============================================================
        
        # Path to main directory (the simulation directory)
        self.sim_path = cfm.SIM_PATH.format(self.SIM_ID)
        
        # List of all directory paths
        self.dir_tree = (
            # LEVEL-0
            self.sim_path,
            # LEVEL-1
            cfm.VIS_PATH.format(self.SIM_ID),
            # LEVEL-2
            cfm.PNG_PATH.format(self.SIM_ID),
            cfm.MP4_PATH.format(self.SIM_ID),
            cfm.BLENDER_PATH.format(self.SIM_ID)
            )
        
        # Other paths
        self.settings_path = cfm.SETTINGS_PATH.format(self.SIM_ID)
        self.blender_path = cfm.BLENDER_PATH.format(self.SIM_ID)
        self.png_path = cfm.PNG_PATH.format(self.SIM_ID)
        self.mp4_path = cfm.MP4_PATH.format(self.SIM_ID)
        
        # Main data file (HDF5)
        self.data_path = cfm.DATA_PATH.format(self.SIM_ID)
        self.dfile: h5py.File = None
        if not self.new_sim:
            self._open_datafile()
        
        # Datasets  ---
        
        # Datasets with shape=(STEPS, NDIM, NUM_PARTICLES)
        self.rcoord_dsname = cfm.RCOORDS_DSNAME
        self.vcoord_dsname = cfm.VCOORDS_DSNAME
        self.coord_dsnames = [self.rcoord_dsname, self.vcoord_dsname]
        
        # Datasets with shape=(STEPS,)
        self.potenergy_dsname = cfm.POTENERGY_DSNAME
        self.kinenergy_dsname = cfm.KINENERGY_DSNAME
        self.temp_dsname = cfm.TEMP_DSNAME
        self.oned_dsnames = [self.potenergy_dsname, self.kinenergy_dsname,
                             self.temp_dsname]
        
        # Lists of all dataset-names
        self.all_dsnames = [*self.coord_dsnames, *self.oned_dsnames]
    
    # ==================================================================
    # ---  Private methods (File management methods)
    # ==================================================================
    
    def _create_tree(self) -> None:
        """
        Creates all the nessessary directories for the simulation.

        If it doesn't yet exist, it also creates a 'simulations-folder'.

        """
        
        # Check if the main simulations-folder already exists
        if not os.path.exists(DIR_SIMFILES):
            os.mkdir(DIR_SIMFILES)
            log.debug("Couldn't find a 'simulations-folder'. "
                      "A new one was created: %s", DIR_SIMFILES)
        
        # Create simulation directory tree
        for directory in self.dir_tree:
            if os.path.exists(directory):
                log.warning("Directory exists and will be overwritten: %s",
                            directory)
                shutil.rmtree(directory)
            os.mkdir(directory)
        
        log.debug('New directory tree created: %s', self.sim_path)
    
    def _open_datafile(self, mode: str = 'a') -> h5py.File:
        """ Open data file. """
        # TODO: Do we need to log this?
        # log.debug("Opening data file in '%s' mode: %s", mode, self.data_path)
        self.dfile = h5py.File(self.data_path, mode=mode)
        return self.dfile
    
    def _close_datafile(self):
        """ Close data file. """
        try:
            self.dfile.close()
            # TODO: Do we need to log this?
            # log.debug("Data file closed: %s", self.data_path)
        except AttributeError:
            log.warning("You tried to close a data file "
                        "(%s) that was already closed.", self.data_path)
    
    def _init_datasets(self) -> None:
        """ Create datasets for a given h5py.File. """
        
        if self.dfile is None:
            raise FileNotFoundError(f"Datafile does not exist! "
                                    f"Expected at: {self.data_path}")
        
        # Datasets with shape=(STEPS, NDIM, NUM_PARTICLES)
        for name in self.coord_dsnames:
            self.dfile.create_dataset(name, dtype=DTYPE,
                                      shape=(self.STEPS,
                                             self.NDIM, self.NUM_PARTICLES))
        
        # Datasets with shape=(STEPS,)
        for name in self.oned_dsnames:
            self.dfile.create_dataset(name, dtype=DTYPE,
                                      shape=(self.STEPS,))
        
        log.debug('HDF datasets initiated/created.')
    
    def _save_settings(self) -> None:
        """ Save system settings to pickle file. """
        with open(self.settings_path, 'wb') as f:
            pickle.dump(self._settings, f, pickle.HIGHEST_PROTOCOL)
    
    # ==================================================================
    # ---  Class methods
    # ==================================================================
    
    @classmethod
    def new(cls,
            simid: Optional[str] = None,
            initfilesys: bool = True,
            **kwargs) -> 'SysManager':
        """ TODO: missing doc string """
        if simid is None:
            simid = _gen_newid()
        path_sim = cfm.SIM_PATH.format(simid)
        records = {'id': simid,
                   'path': path_sim}
        log.info('New simulation: %s', records)
        sm = cls(new_sim=True, simid=simid, **kwargs)
        if initfilesys:
            sm.initfilesys()
        return sm
    
    @classmethod
    def load(cls, simid: str) -> 'SysManager':
        """ TODO: missing doc string """
        path_sim = cfm.SIM_PATH.format(simid)
        path_settings = cfm.SETTINGS_PATH.format(simid)
        paths_exist = (os.path.exists(path_sim), os.path.exists(path_settings))
        if not all(paths_exist):
            records = {'id': simid,
                       'path': path_sim,
                       'settings': path_settings}
            raise FileNotFoundError(f"Couldn't load simulation: {records}")
        log.info("Loading simulation: %s", simid)
        with open(path_settings, 'rb') as f:
            kwargs: dict = pickle.load(f)
        kwargs.pop('new_sim')  # we're overwritting this!
        return cls(new_sim=False, **kwargs)
    
    # ==================================================================
    # ---  Public/user methods (File management methods)
    # ==================================================================
    
    def initfilesys(self) -> None:
        """ Sets up the whole file-system for you. """
        
        if not self.new_sim:
            records = {'sim ID': self.SIM_ID,
                       'new simulation': self.new_sim}
            msg = f"You were about to initialize an existing simulation, " \
                  f"which is not allowed. " \
                  f"Try loading this simulation: {records}"
            raise SystemExit(msg)
        
        # Set-up the simulation directory
        self._create_tree()
        self._save_settings()
        
        # Set-up the datafile and it's datasets
        self._open_datafile()
        self._init_datasets()
        # Remember to close the datafile
        self._close_datafile()
    
    def updateds(self, dsname: str,
                 x: Union[list, np.ndarray],
                 i: int) -> None:
        """ Update values of a dataset for a specific index. """
        
        try:
            # Update datafile's dataset's i^th entry
            self.dfile[dsname][i] = x
        
        except TypeError as e:
            if self.dfile is None:
                msg = "The dataset's datafile you are trying to update " \
                      "might be closed or not exist."
                raise FileNotFoundError(msg)
            raise e
        
        except ValueError as e:
            if self.dfile.id.valid:
                # if file already exists: raise the exception
                raise ValueError(f"Couldn't update dataset: {e}")
            msg = "A ValueError ('%s') was catched while trying to update a " \
                  "dataset. However, we noticed that the datafile in " \
                  "question wasn't open. We'll try to fix this and retry..."
            log.warning(msg, e)
            self._open_datafile()
            self.updateds(dsname, x, i)
        
        except Exception as e:
            self._close_datafile()
            raise e
    
    def getds(self, dsname: str) -> np.ndarray:
        """ Retrieve data from a specific dataset. """
        try:
            return self.dfile[dsname][:]
        except Exception as e:
            self._close_datafile()
            raise e
    
    def delete(self) -> None:
        """ Delete file system. Deletes the entire directory tree. """
        shutil.rmtree(self.sim_path, ignore_errors=True)
        log.info('Directory tree deleted: %s', self.sim_path)
