import tarfile
import time
import urllib.request
from subprocess import Popen, PIPE, STDOUT, run
import os

import pytest
from elasticsearch import Elasticsearch

from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader

from haystack.database.sql import SQLDocumentStore
from haystack.database.memory import InMemoryDocumentStore
from haystack.database.elasticsearch import ElasticsearchDocumentStore

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
def test_docs_xs():
    return [
        {"name": "filename1", "text": "My name is Carla and I live in Berlin", "meta": {"meta_field": "test1"}},
        {"name": "filename2", "text": "My name is Paul and I live in New York", "meta": {"meta_field": "test2"}},
        {"name": "filename3", "text": "My name is Christelle and I live in Paris", "meta": {"meta_field": "test3"}}
    ]


@pytest.fixture()
def farm_reader():
    return FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)


@pytest.fixture()
def transformers_reader():
    return TransformersReader(model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)


@pytest.fixture(params=["farm", "transformers"])
def reader(request):
    if request.param == "farm":
        return FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)
    if request.param == "transformers":
        return TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                  tokenizer="distilbert-base-uncased",
                                  use_gpu=-1)


@pytest.fixture(params=["sql", "memory", "elasticsearch"])
def document_store_with_docs(request, test_docs_xs):
    if request.param == "sql":
        if os.path.exists("qa_test.db"):
            os.remove("qa_test.db")
        document_store = SQLDocumentStore(url="sqlite:///qa_test.db")

    if request.param == "memory":
        document_store = InMemoryDocumentStore()

    if request.param == "elasticsearch":
        document_store = ElasticsearchDocumentStore()

    document_store.write_documents(test_docs_xs)
    return document_store
