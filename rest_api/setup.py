import logging
from pathlib import Path

from setuptools import setup, find_packages


VERSION = "0.0.0"
try:
    VERSION = open(Path(__file__).parent.parent / "VERSION.txt", "r").read()
except Exception as e:
    logging.exception("No VERSION.txt found!")


setup(
    name="farm-haystack-rest-api",
    version=VERSION,
    description="Demo REST API server for Haystack (https://github.com/deepset-ai/haystack)",
    author="deepset.ai",
    author_email="malte.pietsch@deepset.ai",
    url=" https://github.com/deepset-ai/haystack/tree/master/rest_api",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=[
        # The link below cannot be translated properly into setup.cfg
        # because it looks into the parent folder.
        # TODO check if this is still a limitation later on
        f"farm-haystack @ file://localhost/{Path(__file__).parent.parent}#egg=farm-haystack",
        "fastapi<1",
        "uvicorn<1",
        "gunicorn<21",
        "python-multipart<1",  # optional FastAPI dependency for form data
    ],
)
