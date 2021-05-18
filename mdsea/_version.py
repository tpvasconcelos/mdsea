import sys

if sys.version_info >= (3, 8):
    # noinspection PyCompatibility
    from importlib.metadata import version
else:
    from importlib_metadata import version

__version__ = version("mdsea")
