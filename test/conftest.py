import os
import subprocess
import time
from subprocess import run
from sys import platform

import pytest
import requests
from elasticsearch import Elasticsearch
from haystack.retriever.sparse import ElasticsearchFilterOnlyRetriever, ElasticsearchRetriever, TfidfRetriever

from haystack.retriever.dense import DensePassageRetriever, EmbeddingRetriever

from haystack import Document
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader


@pytest.fixture(scope="session")
def elasticsearch_fixture():
    # test if a ES cluster is already running. If not, download and start an ES instance locally.
    try:
        client = Elasticsearch(hosts=[{"host": "localhost", "port": "9200"}])
        client.info()
    except:
        print("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker rm haystack_test_elastic'],
            shell=True
        )
        status = subprocess.run(
            ['docker run -d --name haystack_test_elastic -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.1'],
            shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Elasticsearch. Please check docker container logs.")
        time.sleep(30)


@pytest.fixture(scope="session")
def tika_fixture():
    try:
        tika_url = "http://localhost:9998/tika"
        ping = requests.get(tika_url)
        if ping.status_code != 200:
            raise Exception(
                "Unable to connect Tika. Please check tika endpoint {0}.".format(tika_url))
    except:
        print("Starting Tika ...")
        status = subprocess.run(
            ['docker run -d --name tika -p 9998:9998 apache/tika:1.24.1'],
            shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Tika. Please check docker container logs.")
        time.sleep(30)


@pytest.fixture(scope="session")
def xpdf_fixture(tika_fixture):
    verify_installation = run(["pdftotext"], shell=True)
    if verify_installation.returncode == 127:
        if platform.startswith("linux"):
            platform_id = "linux"
            sudo_prefix = "sudo"
        elif platform.startswith("darwin"):
            platform_id = "mac"
            # For Mac, generally sudo need password in interactive console.
            # But most of the cases current user already have permission to copy to /user/local/bin.
            # Hence removing sudo requirement for Mac.
            sudo_prefix = ""
        else:
            raise Exception(
                """Currently auto installation of pdftotext is not supported on {0} platform """.format(platform)
            )

        commands = """ wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-{0}-4.02.tar.gz &&
                       tar -xvf xpdf-tools-{0}-4.02.tar.gz &&
                       {1} cp xpdf-tools-{0}-4.02/bin64/pdftotext /usr/local/bin""".format(platform_id, sudo_prefix)
        run([commands], shell=True)

        verify_installation = run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                 You can download for your OS from here: https://www.xpdfreader.com/download.html."""
            )


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "sql"])
def document_store(request, test_docs_xs, elasticsearch_fixture):
    return get_document_store(request.param)


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
                                  use_gpu=-1, top_k_per_candidate=5)


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


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "sql"])
def document_store_with_docs(request, test_docs_xs, elasticsearch_fixture):
    document_store = get_document_store(request.param)
    document_store.write_documents(test_docs_xs)
    return document_store


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "sql"])
def document_store(request, test_docs_xs, elasticsearch_fixture):
    return get_document_store(request.param)


@pytest.fixture(params=["es_filter_only", "elsticsearch", "dpr", "embedded", "tfid"])
def retriever(request, document_store):
    return get_retriever(request.param, document_store)


@pytest.fixture(params=["es_filter_only", "elsticsearch", "dpr", "embedded", "tfid"])
def retriever_with_docs(request, document_store_with_docs):
    return get_retriever(request.param, document_store_with_docs)


def get_document_store(document_store_type):
    if document_store_type == "sql":
        if os.path.exists("haystack_test.db"):
            os.remove("haystack_test.db")
        document_store = SQLDocumentStore(url="sqlite:///haystack_test.db")
    elif document_store_type == "memory":
        document_store = InMemoryDocumentStore()
    elif document_store_type == "elasticsearch":
        # make sure we start from a fresh index
        client = Elasticsearch()
        client.indices.delete(index='haystack_test*', ignore=[404])
        document_store = ElasticsearchDocumentStore(index="haystack_test")
    elif document_store_type == "faiss":
        if os.path.exists("haystack_test_faiss.db"):
            os.remove("haystack_test_faiss.db")
        document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db")
    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")

    return document_store


def get_retriever(retriever_type, document_store):

    if retriever_type == "dpr":
        retriever = DensePassageRetriever(document_store=document_store,
                                          query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                          passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                          use_gpu=False, embed_title=True,
                                          remove_sep_tok_from_untitled_passages=True)
    elif retriever_type == "tfid":
        return TfidfRetriever(document_store=document_store)
    elif retriever_type == "embedded":
        retriever = EmbeddingRetriever(document_store=document_store,
                                       embedding_model="deepset/sentence_bert",
                                       use_gpu=False)
    elif retriever_type == "elsticsearch":
        retriever = ElasticsearchRetriever(document_store=document_store)
    elif retriever_type == "es_filter_only":
        retriever = ElasticsearchFilterOnlyRetriever(document_store=document_store)
    else:
        raise Exception(f"No retriever fixture for '{retriever_type}'")

    return retriever
