#!/usr/local/bin/python
# coding: utf-8
"""
Remove the entire simulation directory.

"""
from tests import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")
sm.delete()
