import os
import re
from io import open
from typing import List, Dict

from setuptools import find_packages, setup

# 1. all packages should be listed here with their version requirements if any
packages: List[str] = [
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
    "selenium",
    "webdriver-manager",
    "beautifulsoup4",
    "markdown",
    "streamlit",
]
# this is a lookup table with items like:
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
dependencies: Dict = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in packages)}


def dependencies_list(*package_list):
    """
    Return package with version from list
    """
    return [dependencies[package] for package in package_list]


install_requires = [
    dependencies["farm"],
    dependencies["torch"],
    dependencies["fastapi"],
    dependencies["uvicorn"],
    dependencies["gunicorn"],
    dependencies["pandas"],
    dependencies["sklearn"],
    dependencies["psycopg2-binary"] + "; sys_platform != 'win32' and sys_platform != 'cygwin'",
    dependencies["mmh3"],
    dependencies["sqlalchemy"],
    dependencies["sqlalchemy_utils"],
    dependencies["networkx"],
    dependencies["uvloop"] + "; sys_platform != 'win32' and sys_platform != 'cygwin'",
]

extras_require = {}
extras_require["elasticsearch"] = dependencies_list("elasticsearch")
extras_require["faiss-cpu"] = dependencies_list("faiss-cpu")
extras_require["faiss-gpu"] = dependencies_list("faiss-gpu")
extras_require["weaviate"] = dependencies_list("weaviate-client")
extras_require["ui"] = dependencies_list("streamlit")
extras_require["crawling"] = dependencies_list("selenium", "webdriver-manager")
extras_require["testing"] = dependencies_list("pytest", "mypy","tox")
extras_require["milvus"] = dependencies_list("pymilvus")
extras_require["sentence-transformers"] = dependencies_list("sentence-transformers")
extras_require["all"] = (
        extras_require["elasticsearch"]
        + extras_require["faiss-gpu"]
        + extras_require["weaviate"]
        + extras_require["ui"]
        + extras_require["crawling"]
        + extras_require["testing"]
        + extras_require["milvus"]
        + extras_require["sentence-transformers"]
)


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
    extras_require=extras_require,
    python_requires=">=3.7.0",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
