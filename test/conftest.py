import os
import tarfile
import time
import urllib.request
from subprocess import Popen, PIPE, STDOUT, run

import pytest
from elasticsearch import Elasticsearch

from haystack.database.base import Document
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.database.memory import InMemoryDocumentStore
from haystack.database.sql import SQLDocumentStore
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
        print("Downloading and starting an Elasticsearch instance for the tests ...")
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
        # current "dict" format for a document
        {"text": "My name is Carla and I live in Berlin", "meta": {"meta_field": "test1", "name": "filename1"}},
        # meta_field at the top level for backward compatibility
        {"text": "My name is Paul and I live in New York", "meta_field": "test2", "name": "filename2"},
        # Document object for a doc
        Document(text="My name is Christelle and I live in Paris", meta={"meta_field": "test3", "name": "filename3"})
    ]


@pytest.fixture(params=["farm", "transformers"])
def reader(request):
    if request.param == "farm":
        return FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad",
                          use_gpu=False, top_k_per_sample=5, num_processes=0)
    if request.param == "transformers":
        return TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                  tokenizer="distilbert-base-uncased",
                                  use_gpu=-1)


# TODO Fix bug in test_no_answer_output when using
# @pytest.fixture(params=["farm", "transformers"])
@pytest.fixture(params=["farm"])
def no_answer_reader(request):
    if request.param == "farm":
        return FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                          use_gpu=False, top_k_per_sample=5, no_ans_boost=0, num_processes=0)
    if request.param == "transformers":
        return TransformersReader(model="deepset/roberta-base-squad2",
                                  tokenizer="deepset/roberta-base-squad2",
                                  use_gpu=-1, n_best_per_passage=5)


@pytest.fixture()
def prediction(reader, test_docs_xs):
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]
    prediction = reader.predict(question="Who lives in Berlin?", documents=docs, top_k=5)
    return prediction


@pytest.fixture()
def no_answer_prediction(no_answer_reader, test_docs_xs):
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]
    prediction = no_answer_reader.predict(question="What is the meaning of life?", documents=docs, top_k=5)
    return prediction


@pytest.fixture(params=["sql", "memory", "elasticsearch"])
def document_store_with_docs(request, test_docs_xs, elasticsearch_fixture):
    if request.param == "sql":
        if os.path.exists("qa_test.db"):
            os.remove("qa_test.db")
        document_store = SQLDocumentStore(url="sqlite:///qa_test.db")
        document_store.write_documents(test_docs_xs)

    if request.param == "memory":
        document_store = InMemoryDocumentStore()
        document_store.write_documents(test_docs_xs)

    if request.param == "elasticsearch":
        # make sure we start from a fresh index
        client = Elasticsearch()
        client.indices.delete(index='haystack_test', ignore=[404])
        document_store = ElasticsearchDocumentStore(index="haystack_test")
        assert document_store.get_document_count() == 0
        document_store.write_documents(test_docs_xs)

    return document_store


@pytest.fixture(params=["sql", "memory", "elasticsearch"])
def document_store(request, test_docs_xs, elasticsearch_fixture):
    if request.param == "sql":
        if os.path.exists("qa_test.db"):
            os.remove("qa_test.db")
        document_store = SQLDocumentStore(url="sqlite:///qa_test.db")

    if request.param == "memory":
        document_store = InMemoryDocumentStore()

    if request.param == "elasticsearch":
        # make sure we start from a fresh index
        client = Elasticsearch()
        client.indices.delete(index='haystack_test', ignore=[404])
        document_store = ElasticsearchDocumentStore(index="haystack_test")

    return document_store
