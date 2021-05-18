"""
Remove the entire simulation directory.

"""
import mdsea as md

sm = md.SysManager.load(simid="_mdsea_docs_example")
sm.delete()
