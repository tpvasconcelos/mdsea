#!/usr/local/bin/python
# coding: utf-8
from mdsea.vis.mpl import MPL
import mdsea as md

sm = md.SysManager.load(simid="_mdsea_testsimulation")

mpl = MPL(sm)
mpl.plt_energies()
mpl.plt_temp()
# mpl.plt_rdf()
mpl.plt_sd()
