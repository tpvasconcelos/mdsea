import cProfile
import pstats

from mdsea.helpers import setup_logging

setup_logging(level="DEBUG")


def main():
    from examples import vis_blender

    vis_blender.main()


cProfile.run("main()", "blender.pstat")

p = pstats.Stats("blender.pstat")
p.sort_stats("cumulative").print_stats(20)
