import cProfile
import pstats


def main():
    from examples import vis_blender

    vis_blender.main()


cProfile.run("main()", "blender.pstat")

p = pstats.Stats("blender.pstat")
p.sort_stats("cumulative").print_stats(20)
