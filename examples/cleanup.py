#!/usr/local/bin/python
# coding: utf-8
"""
Remove the entire simulation directory.

"""
import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")
sm.delete()
