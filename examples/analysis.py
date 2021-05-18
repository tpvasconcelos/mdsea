import mdsea as md
from mdsea.vis.matplotlib import MPL

sm = md.SysManager.load(simid="_mdsea_docs_example")

mpl = MPL(sm)
mpl.plt_energies()
mpl.plt_temp()
# mpl.plt_rdf()
mpl.plt_sd()
