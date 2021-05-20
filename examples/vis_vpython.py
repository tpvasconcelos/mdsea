from mdsea.core import SysManager
from mdsea.helpers import setup_logging
from mdsea.vis.vpython import VpythonAnimation

setup_logging(level="DEBUG")

sm = SysManager.load(simid="_mdsea_docs_example")

anim = VpythonAnimation(sm)
anim.run()
