from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from mdsea.f_constants import physical_constants


@dataclass(unsafe_hash=False)
class _Config:
    # Physical constants
    k_boltzmann: float = field(default=physical_constants["k_boltzmann"])
    gravity_acceleration: float = field(default=physical_constants["gravity_acceleration"])

    def update(self, new: dict) -> None:
        for key, value in new.items():
            if not hasattr(self, key):
                raise AttributeError(f"Not a valid config field: {key}")
            setattr(self, key, value)


Config = _Config()


@contextmanager
def config_context(**new_config):
    """** EXPERIMENTAL FEATURE **

    Inspired by sklearn's ``config_context``

    """
    old_config = asdict(Config)
    Config.update(new=new_config)
    try:
        yield
    finally:
        Config.update(new=old_config)
