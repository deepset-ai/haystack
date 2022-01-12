import os
from setuptools import setup


def get_version() -> str:
    """
    Read the version from haystack/_version.py without importing haystack.
    """
    path = os.path.join(os.path.dirname(__file__), "haystack/_version.py")
    path = os.path.normpath(os.path.abspath(path))
    with open(path) as f:
        for line in f:
            if line.startswith("__version__"):
                _, version = line.split(" = ", 1)
                version = version.replace("\"", "").strip()
                return version
    raise RuntimeError("Unable to find version string in {}.".format(path))


if __name__ == '__main__':
    setup(
        name="haystack",
        version=get_version()
    )