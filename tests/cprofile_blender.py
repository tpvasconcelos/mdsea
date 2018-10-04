#!/usr/local/bin/python
# coding: utf-8
import cProfile
import pstats


def main():
    from tests import vis_blender
    vis_blender.main()


cProfile.run("main()", "blender.pstat")

p = pstats.Stats("blender.pstat")
p.sort_stats("cumulative").print_stats(20)
