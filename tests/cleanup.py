#!/usr/local/bin/python
# coding: utf-8
from tests import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")

# Remove the entire simulation directory.
sm.delete()
