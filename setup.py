import os
import re
from io import open

from setuptools import find_packages, setup

# IMPORTANT:
# 1. all packages should be listed here with their version requirements if any
packages = [
    "farm==0.8.0",
    "torch",  # torch package version already mentioned in farm
    "fastapi",
    "uvicorn",
    "gunicorn",
    "pandas",
    "sklearn",
    "psycopg2-binary",
    "elasticsearch>=7.7,<=7.10",
    "elastic-apm",
    "tox",
    "coverage",
    "langdetect",  # for PDF conversions
    "sentence-transformers",
    "python-multipart",
    "python-docx",
    "sqlalchemy>=1.4.2",
    "sqlalchemy_utils",
    "faiss-cpu>=1.6.3",
    "faiss-gpu",
    "tika",
    "uvloop==0.14",
    "httptools",
    "nltk",
    "more_itertools",
    "networkx",
    "pymilvus",
    "SPARQLWrapper",
    "mmh3",
    "weaviate-client",
    "mypy",
    "pytest",
    "sentence-transformers",
    "selenium",
    "webdriver-manager",
    "beautifulsoup4",
    "markdown",
    "streamlit",
]
# this is a lookup table with items like:
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in packages)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


install_requires = [
    deps["farm"],
    deps["torch"],
    deps["fastapi"],
    deps["uvicorn"],
    deps["gunicorn"],
    deps["pandas"],
    deps["sklearn"],
    deps["psycopg2-binary"] + "; sys_platform != 'win32' and sys_platform != 'cygwin'",
    deps["mmh3"],
    deps["sqlalchemy"],
    deps["sqlalchemy_utils"],
    deps["networkx"],
    deps["uvloop"] + "; sys_platform != 'win32' and sys_platform != 'cygwin'",
]

extras = {}
extras["elasticsearch"] = deps_list("elasticsearch")
extras["faiss-cpu"] = deps_list("faiss-cpu")
extras["faiss-gpu"] = deps_list("faiss-gpu")
extras["weaviate"] = deps_list("weaviate-client")
extras["ui"] = deps_list("streamlit")
extras["crawer"] = deps_list("selenium", "webdriver-manager")
extras["testing"] = deps_list("pytest", "mypy")


def versionfromfile(*filepath):
    infile = os.path.join(*filepath)
    with open(infile) as fp:
        version_match = re.search(
                r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string in {}.".format(infile))


here = os.path.abspath(os.path.dirname(__file__))
_version: str = versionfromfile(here, "haystack", "_version.py")

setup(
    name="farm-haystack",
    version=_version,
    author="Malte Pietsch, Timo Moeller, Branden Chan, Tanay Soni",
    author_email="malte.pietsch@deepset.ai",
    description="Neural Question Answering & Semantic Search at Scale. Use modern transformer based models like BERT to find answers in large document collections",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="QA Question-Answering Reader Retriever semantic-search search BERT roberta albert squad mrc transfer-learning language-model transformer",
    license="Apache",
    url="https://github.com/deepset-ai/haystack",
    download_url=f"https://github.com/deepset-ai/haystack/archive/{_version}.tar.gz",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requires,
    extras_require=extras,
    python_requires=">=3.7.0",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
