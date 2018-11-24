#!/usr/local/bin/python
# coding: utf-8
import logging
import math
import statistics as stats
from time import time
from typing import Any, Callable, Optional, Sequence, Tuple

from scipy.special import gamma

from mdsea import loghandler

log = logging.getLogger(__name__)
log.addHandler(loghandler)


##########
#  USED  #
##########

def nsphere_volume(n, r):
    return (math.pi ** (n / 2)) * (r ** n) / gamma(n / 2 + 1)


def rgb2unit(rgb: tuple) -> tuple:
    assert len(rgb) in (3, 4)
    if len(rgb) == 3:
        return tuple(c / 255 for c in rgb)
    return tuple(c / 255 if i < 3 else c for i, c in enumerate(rgb))


def check_val(name: str, val: Any, options: Tuple[Any, ...]) -> None:
    """ Check variable value. """
    if val not in options:
        msg = f"Illegal value ({val}) for {name}. " \
            f"It should be one of: {options}"
        raise ValueError(msg)


def check_type(name: str, val: Any, target_types: Tuple[type, ...]) -> None:
    """ Check variable type. """
    if type(val) not in target_types:
        msg = f"Illegal type ({type(val)}) for {name}. " \
            f"It should be one of: {target_types}"
        raise TypeError(msg)


def check_size(names: Tuple[str, ...], vals: Tuple[Sequence, ...]) -> None:
    """ Check maching sizes. """
    l1 = len(vals[0])
    for v in vals:
        if len(v) != l1:
            msg = "The following variables all " \
                f"need to be the same lenght: {names}"
            raise ValueError(msg)


def get_dt(radius: float, mean_speed: float,
           drpf: float = 0.01) -> float:
    """
    Generate a delta time (dt, time-step, etc) based on a
    given 'displacement ratio per frame' (defined bellow).
    
    @:param drpf "Displacement ratio per frame"
    
    """
    if mean_speed == 0:
        mean_speed = 1
        return drpf * radius / mean_speed
    return drpf * radius / mean_speed


# def get_memory() -> dict:
#     """ Returns full memory info. """
#     import psutil
#     proc = psutil.Process(os.getpid())
#     return proc.memory_info()


class ProgressBar:
    
    def __init__(self, name: str, stop: int,
                 loggername: str = __name__,
                 step: int = 10) -> None:
        
        self.name = name
        self.stop = stop
        self.step = step
        self.log = logging.getLogger(loggername)
        
        self.prefix = f"ProgresBar ({self.name}):"
        
        self.progress = 0
        self.timespercycle = []
        self.t_lastcycle = time()
        
        self.tstart = time()
        self.tfinish = None
    
    def set_start(self, t: Optional[float] = None) -> None:
        if not t:
            t = time()
        self.tstart = t
        self.log.debug("%s %s", self.prefix, {'start-time': self.tstart})
    
    def set_finish(self, tfinish: Optional[float] = None) -> None:
        if not tfinish:
            tfinish = time()
        self.tfinish = tfinish
        self.log.debug("%s %s", self.prefix, {'end-time': self.tfinish})
    
    def get_duration(self, ndigits: int = 1) -> float:
        if self.tfinish is None:
            self.log.warning(
                "You forgot to finish the '%s' ProgressBar! "
                "Setting the finished time now.", self.name)
            self.set_finish()
        return float(round(self.tfinish - self.tstart, ndigits=ndigits))
    
    def log_duration(self, ndigits: int = 1) -> None:
        records = {'status': 'finished',
                   'lifetime': f'{self.get_duration(ndigits)}s'}
        self.log.info("%s %s", self.prefix, records)
    
    def log_progress(self, i) -> None:
        
        # Evaluate progress percentage
        percent = round(100 * (i + 1) / self.stop)
        
        # Round percentage to step
        rpercent = int(self.step * round(float(percent) / self.step))
        if self.progress >= rpercent or rpercent % self.step:
            return
        
        stepsleft = (100 - percent) / self.step
        
        if i <= 1:
            # Ignore the first longer setup cycle
            eta = round(stepsleft * (time() - self.t_lastcycle))
        else:
            self.timespercycle.append(time() - self.t_lastcycle)
            eta = round(stepsleft * stats.mean(self.timespercycle))
        
        records = {'step': f'{i + 1}/{self.stop}',
                   'percentage': percent,
                   'ETA': f'{eta}s'}
        
        self.log.info("%s %s", self.prefix, records)
        
        self.progress = rpercent
        self.t_lastcycle = time()


# ----------------------------------------------------------------------


##############
#  NOT USED  #
##############

def nd_spherical_coords(i, ndim):
    if i == ndim - 1:
        return f"x_{i} = r * cos(φ_{ndim - 2}) " \
               + "".join(f"* sin(φ_{i})" for i in range(ndim - 2))
    
    elif i == ndim - 2:
        return f"x_{i} = r * sin(φ_{ndim - 2}) " \
               + "".join(f"* sin(φ_{i})" for i in range(ndim - 2))
    elif i == 0:
        return f"x_{i} = r * cos(φ_0)"
    else:
        return f"x_{i} = r * cos(φ_{i}) " \
               + "".join(f"* sin(φ_{i})" for i in range(i))


def timethis(method: Callable) -> Any:
    from time import time
    
    def timed_method(*args, **kw):
        t_start = time()
        result = method(*args, **kw)
        print('%r (%r, %r) {} sec'
              .format(method.__name__, args, kw, round(time() - t_start, 2)))
        return result
    
    return timed_method


def lastsim_path(simsdir) -> str:
    """ Returns the path to the most recent simulation directory. """
    import os
    all_sim_dirs = [d for d in os.listdir(simsdir) if d.isdigit()]
    if len(all_sim_dirs):
        most_recent_id = max(int(directory) for directory in all_sim_dirs)
        return f'{simsdir}/{str(most_recent_id)}'
    else:
        msg = f"{simsdir} does not contain any simulation directories!"
        raise SystemError(msg)


def print_title_box(title: str,
                    upcase: bool = True) -> str:
    hash_bar = (len(title) + 6) * '#'
    if upcase:
        title = title.upper()
    title_box = f"{hash_bar}\n#  {title}  #\n{hash_bar}"
    print(title_box)
    return title_box


if __name__ == '__main__':
    # print_title_box("minlevel settings", upcase=False)
    N = 5
    for i in range(N):
        print(f"{nd_spherical_coords(i, N)}")
