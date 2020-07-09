import tarfile
import time
import urllib.request
from subprocess import Popen, PIPE, STDOUT, run

import pytest
from elasticsearch import Elasticsearch

from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader

@pytest.fixture(scope='session')
def elasticsearch_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('elasticsearch')


@pytest.fixture(scope="session")
def elasticsearch_fixture(elasticsearch_dir):
    # test if a ES cluster is already running. If not, download and start an ES instance locally.
    try:
        client = Elasticsearch(hosts=[{"host": "localhost"}])
        client.info()
    except:
        thetarfile = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.1-linux-x86_64.tar.gz"
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall(path=elasticsearch_dir)
        es_server = Popen([elasticsearch_dir / "elasticsearch-7.6.1/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT)
        time.sleep(40)


@pytest.fixture(scope="session")
def xpdf_fixture():
    verify_installation = run(["pdftotext"], shell=True)
    if verify_installation.returncode == 127:
        commands = """ wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.02.tar.gz &&
                       tar -xvf xpdf-tools-linux-4.02.tar.gz && sudo cp xpdf-tools-linux-4.02/bin64/pdftotext /usr/local/bin"""
        run([commands], shell=True)

        verify_installation = run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                 You can download for your OS from here: https://www.xpdfreader.com/download.html."""
            )


@pytest.fixture()
def farm_reader():
    return FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)

@pytest.fixture()
def transformers_reader():
    return TransformersReader(model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=False)

