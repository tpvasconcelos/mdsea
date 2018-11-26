#!/usr/local/bin/python
# coding: utf-8
import logging
import os

import numpy as np

from mdsea import loghandler
from mdsea.core import SysManager
from mdsea.quicker import norm

log = logging.getLogger(__name__)
log.addHandler(loghandler)


def make_mp4(fname: str,
             stills_dir: str,
             fps: float,
             timeit: bool = False) -> None:
    """ Render mp4 video file from image stills. """
    
    if timeit:
        from time import time
        t_start = time()
    
    os.chdir(stills_dir)
    
    settings = {
        "util": "ffmpeg",
        "infile": "-i img%06d.png",
        "video-codec": "-vcodec libx264",
        "bitrate": "-b:v 6000k",
        "frame_rate": f"-r {fps} -framerate {fps}",
        "pixel-format": "-pix_fmt yuv420p",
        "outfile": fname
        }
    
    log.info("Running ffmpeg's stills-to-mp4 conversion...")
    
    # ffmpeg -i img%06d.png -vcodec libx264 -b:v 6000k
    # -r 24 -framerate 24 -pix_fmt yuv420p movie.mp4
    cmd = " ".join(settings.values())
    log.debug(cmd)
    os.system(cmd)
    
    if timeit:
        # noinspection PyUnboundLocalVariable
        log.info("ffmpeg took %ss to run", round(time() - t_start, 2))


class Analyser(object):
    """ TODO: docstring """
    
    def __init__(self, sm: SysManager) -> None:
        self.sm = sm
        
        # TODO(tpvasconcelos) explain the difference between r_coords & r_vecs
        self.r_coords = self.sm.get_ds(self.sm.rcoord_dsname)
        self.r_vecs = np.stack(
            (self.r_coords[:, i] for i in range(self.sm.NDIM)), axis=-1)
        
        # TODO(tpvasconcelos) explain the difference between v_coords & v_vecs
        self.v_coords = self.sm.get_ds(self.sm.vcoord_dsname)
        self.v_vecs = np.stack(
            (self.v_coords[:, i] for i in range(self.sm.NDIM)), axis=-1)
        self.speeds = norm(self.v_vecs, axis=-1)
        self.maxspeed = float(np.mean(self.speeds) + (3 * np.std(self.speeds)))
        
        self.mean_pot_energies = self.sm.get_ds(self.sm.potenergy_dsname)
        self.mean_kin_energies = self.sm.get_ds(self.sm.kinenergy_dsname)
        self.total_energy = self.mean_kin_energies + self.mean_pot_energies
        
        self.temp = self.sm.get_ds(self.sm.temp_dsname)
        
        self.mean_pot_energy = np.mean(self.mean_pot_energies)
        self.mean_kin_energy = np.mean(self.mean_kin_energies)
        self.mean_total_energy = np.mean(self.total_energy)
        self.mean_temp = np.mean(self.temp)


class Vis(Analyser):
    """ TODO: docstring """
    
    def __init__(self, sm: SysManager, frame_step: int = 1) -> None:
        super(Vis, self).__init__(sm)
        
        self.frame_step = frame_step
        
        # Round number of frames down (review this at some point!)
        self.num_frames = int(self.sm.STEPS / self.frame_step)
        
        self.step = 0
