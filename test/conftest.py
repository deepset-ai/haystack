import subprocess
import time
from subprocess import run
from sys import platform

import pytest
import requests
from elasticsearch import Elasticsearch

from haystack.generator.transformers import Seq2SeqGenerator
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph
from milvus import Milvus

import weaviate
from haystack.document_store.weaviate import WeaviateDocumentStore

from haystack.document_store.milvus import MilvusDocumentStore
from haystack.generator.transformers import RAGenerator, RAGeneratorType

from haystack.retriever.sparse import ElasticsearchFilterOnlyRetriever, ElasticsearchRetriever, TfidfRetriever

from haystack.retriever.dense import DensePassageRetriever, EmbeddingRetriever

from haystack import Document
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.summarizer.transformers import TransformersSummarizer
from haystack.translator import TransformersTranslator


def pytest_addoption(parser):
    parser.addoption("--document_store_type", action="store", default="all")


def pytest_generate_tests(metafunc):
    # parametrize document_store fixture if it's in the test function argument list
    # but does not have an explicit parametrize annotation e.g
    # @pytest.mark.parametrize("document_store", ["memory"], indirect=False)
    found_mark_parametrize_document_store = False
    for marker in metafunc.definition.iter_markers('parametrize'):
        if 'document_store' in marker.args[0]:
            found_mark_parametrize_document_store = True
            break

    if 'document_store' in metafunc.fixturenames and not found_mark_parametrize_document_store:
        document_store_type = metafunc.config.option.document_store_type
        if "all" in document_store_type:
            document_store_type = "elasticsearch, faiss, memory, milvus"

        document_store_types = [item.strip() for item in document_store_type.split(",")]
        metafunc.parametrize("document_store", document_store_types, indirect=True)


def _sql_session_rollback(self, attr):
    """
    Inject SQLDocumentStore at runtime to do a session rollback each time it is called. This allows to catch
    errors where an intended operation is still in a transaction, but not committed to the database.
    """
    method = object.__getattribute__(self, attr)
    if callable(method):
        try:
            self.session.rollback()
        except AttributeError:
            pass

    return method


SQLDocumentStore.__getattribute__ = _sql_session_rollback


def pytest_collection_modifyitems(items):
    for item in items:
        if "generator" in item.nodeid:
            item.add_marker(pytest.mark.generator)
        elif "summarizer" in item.nodeid:
            item.add_marker(pytest.mark.summarizer)
        elif "tika" in item.nodeid:
            item.add_marker(pytest.mark.tika)
        elif "elasticsearch" in item.nodeid:
            item.add_marker(pytest.mark.elasticsearch)
        elif "graphdb" in item.nodeid:
            item.add_marker(pytest.mark.graphdb)
        elif "pipeline" in item.nodeid:
            item.add_marker(pytest.mark.pipeline)
        elif "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        elif "weaviate" in item.nodeid:
            item.add_marker(pytest.mark.weaviate)


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
            ['docker run -d --name haystack_test_elastic -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'],
            shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Elasticsearch. Please check docker container logs.")
        time.sleep(30)


@pytest.fixture(scope="session")
def milvus_fixture():
    # test if a Milvus server is already running. If not, start Milvus docker container locally.
    # Make sure you have given > 6GB memory to docker engine
    try:
        milvus_server = Milvus(uri="tcp://localhost:19530", timeout=5, wait_timeout=5)
        milvus_server.server_status(timeout=5)
    except:
        print("Starting Milvus ...")
        status = subprocess.run(['docker run -d --name milvus_cpu_0.10.5 -p 19530:19530 -p 19121:19121 '
                                 'milvusdb/milvus:0.10.5-cpu-d010621-4eda95'], shell=True)
        time.sleep(40)

@pytest.fixture(scope="session")
def weaviate_fixture():
    # test if a Weaviate server is already running. If not, start Weaviate docker container locally.
    # Make sure you have given > 6GB memory to docker engine
    try:
        weaviate_server = weaviate.Client(url='http://localhost:8080', timeout_config=(5, 15))
        weaviate_server.is_ready()
    except:
        print("Starting Weaviate servers ...")
        status = subprocess.run(
            ['docker rm haystack_test_weaviate'],
            shell=True
        )
        status = subprocess.run(
            ['docker run -d --name haystack_test_weaviate -p 8080:8080 semitechnologies/weaviate:1.4.0'],
            shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch Weaviate. Please check docker container logs.")
        time.sleep(60)

@pytest.fixture(scope="session")
def graphdb_fixture():
    # test if a GraphDB instance is already running. If not, download and start a GraphDB instance locally.
    try:
        kg = GraphDBKnowledgeGraph()
        # fail if not running GraphDB
        kg.delete_index()
    except:
        print("Starting GraphDB ...")
        status = subprocess.run(
            ['docker rm haystack_test_graphdb'],
            shell=True
        )
        status = subprocess.run(
            ['docker run -d -p 7200:7200 --name haystack_test_graphdb docker-registry.ontotext.com/graphdb-free:9.4.1-adoptopenjdk11'],
            shell=True
        )
        if status.returncode:
            raise Exception(
                "Failed to launch GraphDB. Please check docker container logs.")
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
        commands = """ wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-{0}-4.03.tar.gz &&
                       tar -xvf xpdf-tools-{0}-4.03.tar.gz &&
                       {1} cp xpdf-tools-{0}-4.03/bin64/pdftotext /usr/local/bin""".format(platform_id, sudo_prefix)
        run([commands], shell=True)

        verify_installation = run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                 You can download for your OS from here: https://www.xpdfreader.com/download.html."""
            )


@pytest.fixture(scope="module")
def rag_generator():
    return RAGenerator(
        model_name_or_path="facebook/rag-token-nq",
        generator_type=RAGeneratorType.TOKEN
    )


@pytest.fixture(scope="module")
def eli5_generator():
    return Seq2SeqGenerator(model_name_or_path="yjernite/bart_eli5")


@pytest.fixture(scope="module")
def summarizer():
    return TransformersSummarizer(
        model_name_or_path="google/pegasus-xsum",
        use_gpu=-1
    )


@pytest.fixture(scope="module")
def en_to_de_translator():
    return TransformersTranslator(
        model_name_or_path="Helsinki-NLP/opus-mt-en-de",
    )


@pytest.fixture(scope="module")
def de_to_en_translator():
    return TransformersTranslator(
        model_name_or_path="Helsinki-NLP/opus-mt-de-en",
    )


@pytest.fixture(scope="module")
def test_docs_xs():
    return [
        # current "dict" format for a document
        {"text": "My name is Carla and I live in Berlin", "meta": {"meta_field": "test1", "name": "filename1"}},
        # meta_field at the top level for backward compatibility
        {"text": "My name is Paul and I live in New York", "meta_field": "test2", "name": "filename2"},
        # Document object for a doc
        Document(text="My name is Christelle and I live in Paris", meta={"meta_field": "test3", "name": "filename3"})
    ]


@pytest.fixture(params=["farm", "transformers"], scope="module")
def reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="distilbert-base-uncased-distilled-squad",
            use_gpu=False,
            top_k_per_sample=5,
            num_processes=0
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="distilbert-base-uncased-distilled-squad",
            tokenizer="distilbert-base-uncased",
            use_gpu=-1
        )


# TODO Fix bug in test_no_answer_output when using
# @pytest.fixture(params=["farm", "transformers"])
@pytest.fixture(params=["farm"], scope="module")
def no_answer_reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="deepset/roberta-base-squad2",
            use_gpu=False,
            top_k_per_sample=5,
            no_ans_boost=0,
            return_no_answer=True,
            num_processes=0
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            use_gpu=-1,
            top_k_per_candidate=5
        )


@pytest.fixture(scope="module")
def prediction(reader, test_docs_xs):
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=5)
    return prediction


@pytest.fixture(scope="module")
def no_answer_prediction(no_answer_reader, test_docs_xs):
    docs = [Document.from_dict(d) if isinstance(d, dict) else d for d in test_docs_xs]
    prediction = no_answer_reader.predict(query="What is the meaning of life?", documents=docs, top_k=5)
    return prediction


@pytest.fixture(params=["es_filter_only", "elasticsearch", "dpr", "embedding", "tfidf"])
def retriever(request, document_store):
    return get_retriever(request.param, document_store)


@pytest.fixture(params=["es_filter_only", "elasticsearch", "dpr", "embedding", "tfidf"])
def retriever_with_docs(request, document_store_with_docs):
    return get_retriever(request.param, document_store_with_docs)


def get_retriever(retriever_type, document_store):

    if retriever_type == "dpr":
        retriever = DensePassageRetriever(document_store=document_store,
                                          query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                          passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                          use_gpu=False, embed_title=True)
    elif retriever_type == "tfidf":
        retriever = TfidfRetriever(document_store=document_store)
        retriever.fit()
    elif retriever_type == "embedding":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="deepset/sentence_bert",
            use_gpu=False
        )
    elif retriever_type == "retribert":
        retriever = EmbeddingRetriever(document_store=document_store,
                                       embedding_model="yjernite/retribert-base-uncased",
                                       model_format="retribert",
                                       use_gpu=False)
    elif retriever_type == "elasticsearch":
        retriever = ElasticsearchRetriever(document_store=document_store)
    elif retriever_type == "es_filter_only":
        retriever = ElasticsearchFilterOnlyRetriever(document_store=document_store)
    else:
        raise Exception(f"No retriever fixture for '{retriever_type}'")

    return retriever


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "sql", "milvus"])
def document_store_with_docs(request, test_docs_xs):
    document_store = get_document_store(request.param)
    document_store.write_documents(test_docs_xs)
    yield document_store
    document_store.delete_all_documents()


@pytest.fixture
def document_store(request, test_docs_xs):
    vector_dim = request.node.get_closest_marker("vector_dim", pytest.mark.vector_dim(768))
    document_store = get_document_store(request.param, vector_dim.args[0])
    yield document_store
    document_store.delete_all_documents()


def get_document_store(document_store_type, embedding_dim=768, embedding_field="embedding"):
    if document_store_type == "sql":
        document_store = SQLDocumentStore(url="sqlite://", index="haystack_test")
    elif document_store_type == "memory":
        document_store = InMemoryDocumentStore(
            return_embedding=True, embedding_dim=embedding_dim, embedding_field=embedding_field, index="haystack_test"
        )
    elif document_store_type == "elasticsearch":
        # make sure we start from a fresh index
        client = Elasticsearch()
        client.indices.delete(index='haystack_test*', ignore=[404])
        document_store = ElasticsearchDocumentStore(
            index="haystack_test", return_embedding=True, embedding_dim=embedding_dim, embedding_field=embedding_field
        )
    elif document_store_type == "faiss":
        document_store = FAISSDocumentStore(
            vector_dim=embedding_dim,
            sql_url="sqlite://",
            return_embedding=True,
            embedding_field=embedding_field,
            index="haystack_test",
        )
        return document_store
    elif document_store_type == "milvus":
        document_store = MilvusDocumentStore(
            vector_dim=embedding_dim,
            sql_url="sqlite://",
            return_embedding=True,
            embedding_field=embedding_field,
            index="haystack_test",
        )
        _, collections = document_store.milvus_server.list_collections()
        for collection in collections:
            if collection.startswith("haystack_test"):
                document_store.milvus_server.drop_collection(collection)
        return document_store
    elif document_store_type == "weaviate":
        document_store = WeaviateDocumentStore(
            weaviate_url="http://localhost:8080",
            index="Haystacktest"
        )
        document_store.weaviate_client.schema.delete_all()
        document_store._create_schema_and_index_if_not_exist()
        return document_store
    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")

    return document_store
