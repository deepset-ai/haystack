# setup.py will still be needed for a while to allow editable installs.
# Check regularly in the future if this is still the case, or it can be safely removed.

# Make sure the correct pip is used. Pip below 21.3 will be stuck in a loop on the self referencing extra_requires
# Note: these two lines are incompatible with the existence of a pyproject.toml file, as it will not involve pip
# in the execution of setup.py and therefore break this check. Re evaluate later if this is still the case, and
# if this check is still needed.
import sys
print(sys.argv)

import pkg_resources
try:
    pkg_resources.require(['pip >= 21.3.0'])
except pkg_resources.VersionConflict as vce:
    raise pkg_resources.VersionConflict("Please upgrade your pip to >= 21.3.1 by running 'pip install --upgrade pip'.") from vce

from setuptools import setup
setup()