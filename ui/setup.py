import logging
from pathlib import Path

from setuptools import setup, find_packages


VERSION = "0.0.0"
try:
    # After git clone, VERSION.txt is in the root folder
    VERSION = open(Path(__file__).parent.parent / "VERSION.txt", "r").read()
except Exception:
    try:
        # In Docker, VERSION.txt is in the same folder
        VERSION = open(Path(__file__).parent / "VERSION.txt", "r").read()
    except Exception as e:
        logging.exception("No VERSION.txt found!")

setup(
    name="farm-haystack-ui",
    version=VERSION,
    description="Demo UI for Haystack (https://github.com/deepset-ai/haystack)",
    author="deepset.ai",
    author_email="malte.pietsch@deepset.ai",
    url=" https://github.com/deepset-ai/haystack/tree/main/ui",
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
    install_requires=["streamlit>=1.9.0, <2", "st-annotated-text>=2.0.0, <3", "markdown>=3.3.4, <4"],
)
