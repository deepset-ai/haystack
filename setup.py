import os

# Opt-out dependencies are regulated with environmental variables
# Example: `HAYSTACK_NO_ELASTICSEARCH=1 pip install farm-haystack` will not install Elasticsearch dependencies
extra = "all"
if os.getenv("HAYSTACK_NO_ELASTICSEARCH"):
    extra = "no_es"

print(extra)

# setup.py will still be needed for a while to allow editable installs (pip < 21.1).
# Check regularly in the future if this is still the case, or it can be safely removed.
from setuptools import setup
setup(extra=extra)