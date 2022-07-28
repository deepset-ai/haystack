from datetime import timedelta
from typing import Any, List, Optional, Dict, Union

import subprocess
from uuid import UUID
import time
from subprocess import run
from sys import platform
import gc
import uuid
import logging
from pathlib import Path
import os

import requests_cache
import responses
from sqlalchemy import create_engine, text
import posthog

import numpy as np
import psutil
import pytest
import requests

from haystack.nodes.base import BaseComponent

try:
    from milvus import Milvus

    milvus1 = True
except ImportError:
    milvus1 = False
    from pymilvus import utility

try:
    from elasticsearch import Elasticsearch
    from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
    import weaviate
    from haystack.document_stores.weaviate import WeaviateDocumentStore
    from haystack.document_stores import MilvusDocumentStore, PineconeDocumentStore
    from haystack.document_stores.graphdb import GraphDBKnowledgeGraph
    from haystack.document_stores.faiss import FAISSDocumentStore
    from haystack.document_stores.sql import SQLDocumentStore

except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed("test", "test", ie)

from haystack.document_stores import BaseDocumentStore, DeepsetCloudDocumentStore, InMemoryDocumentStore

from haystack.nodes import BaseReader, BaseRetriever, OpenAIAnswerGenerator
from haystack.nodes.answer_generator.transformers import Seq2SeqGenerator
from haystack.nodes.answer_generator.transformers import RAGenerator
from haystack.nodes.ranker import SentenceTransformersRanker
from haystack.nodes.document_classifier.transformers import TransformersDocumentClassifier
from haystack.nodes.retriever.sparse import FilterRetriever, BM25Retriever, TfidfRetriever
from haystack.nodes.retriever.dense import (
    DensePassageRetriever,
    EmbeddingRetriever,
    MultihopEmbeddingRetriever,
    TableTextRetriever,
)
from haystack.nodes.reader.farm import FARMReader
from haystack.nodes.reader.transformers import TransformersReader
from haystack.nodes.reader.table import TableReader, RCIReader
from haystack.nodes.summarizer.transformers import TransformersSummarizer
from haystack.nodes.translator import TransformersTranslator
from haystack.nodes.question_generator import QuestionGenerator

from haystack.modeling.infer import Inferencer, QAInferencer

from haystack.schema import Document

from .mocks import pinecone as pinecone_mock


# To manually run the tests with default PostgreSQL instead of SQLite, switch the lines below
SQL_TYPE = "sqlite"
# SQL_TYPE = "postgres"

SAMPLES_PATH = Path(__file__).parent / "samples"

# to run tests against Deepset Cloud set MOCK_DC to False and set the following params
DC_API_ENDPOINT = "https://DC_API/v1"
DC_TEST_INDEX = "document_retrieval_1"
DC_API_KEY = "NO_KEY"
MOCK_DC = True

# Disable telemetry reports when running tests
posthog.disabled = True

# Cache requests (e.g. huggingface model) to circumvent load protection
# See https://requests-cache.readthedocs.io/en/stable/user_guide/filtering.html
requests_cache.install_cache(urls_expire_after={"huggingface.co": timedelta(hours=1), "*": requests_cache.DO_NOT_CACHE})


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


def pytest_collection_modifyitems(config, items):
    # add pytest markers for tests that are not explicitly marked but include some keywords
    name_to_markers = {
        "generator": [pytest.mark.generator],
        "summarizer": [pytest.mark.summarizer],
        "tika": [pytest.mark.tika, pytest.mark.integration],
        "parsr": [pytest.mark.parsr, pytest.mark.integration],
        "ocr": [pytest.mark.ocr, pytest.mark.integration],
        "elasticsearch": [pytest.mark.elasticsearch],
        "faiss": [pytest.mark.faiss],
        "milvus": [pytest.mark.milvus, pytest.mark.milvus1],
        "weaviate": [pytest.mark.weaviate],
        "pinecone": [pytest.mark.pinecone],
        # FIXME GraphDB can't be treated as a regular docstore, it fails most of their tests
        "graphdb": [pytest.mark.integration],
    }
    for item in items:
        for name, markers in name_to_markers.items():
            if name in item.nodeid.lower():
                for marker in markers:
                    item.add_marker(marker)

        # if the cli argument "--document_store_type" is used, we want to skip all tests that have markers of other docstores
        # Example: pytest -v test_document_store.py --document_store_type="memory" => skip all tests marked with "elasticsearch"
        document_store_types_to_run = config.getoption("--document_store_type")
        document_store_types_to_run = [docstore.strip() for docstore in document_store_types_to_run.split(",")]
        keywords = []

        for i in item.keywords:
            if "-" in i:
                keywords.extend(i.split("-"))
            else:
                keywords.append(i)
        for cur_doc_store in [
            "elasticsearch",
            "faiss",
            "sql",
            "memory",
            "milvus1",
            "milvus",
            "weaviate",
            "pinecone",
            "opensearch",
        ]:
            if cur_doc_store in keywords and cur_doc_store not in document_store_types_to_run:
                skip_docstore = pytest.mark.skip(
                    reason=f'{cur_doc_store} is disabled. Enable via pytest --document_store_type="{cur_doc_store}"'
                )
                item.add_marker(skip_docstore)

        if "milvus1" in keywords and not milvus1:
            skip_milvus1 = pytest.mark.skip(reason="Skipping Tests for 'milvus1', as Milvus2 seems to be installed.")
            item.add_marker(skip_milvus1)
        elif "milvus" in keywords and milvus1:
            skip_milvus = pytest.mark.skip(reason="Skipping Tests for 'milvus', as Milvus1 seems to be installed.")
            item.add_marker(skip_milvus)

        # Skip PineconeDocumentStore if PINECONE_API_KEY not in environment variables
        # if not os.environ.get("PINECONE_API_KEY", False) and "pinecone" in keywords:
        #     skip_pinecone = pytest.mark.skip(reason="PINECONE_API_KEY not in environment variables.")
        #     item.add_marker(skip_pinecone)


#
# Empty mocks, as a base for unit tests.
#
# Monkeypatch the methods you need with either a mock implementation
# or a unittest.mock.MagicMock object (https://docs.python.org/3/library/unittest.mock.html)
#


class MockNode(BaseComponent):
    outgoing_edges = 1

    def run(self, *a, **k):
        pass

    def run_batch(self, *a, **k):
        pass


class MockDocumentStore(BaseDocumentStore):
    outgoing_edges = 1

    def _create_document_field_map(self, *a, **k):
        pass

    def delete_documents(self, *a, **k):
        pass

    def delete_labels(self, *a, **k):
        pass

    def get_all_documents(self, *a, **k):
        pass

    def get_all_documents_generator(self, *a, **k):
        pass

    def get_all_labels(self, *a, **k):
        pass

    def get_document_by_id(self, *a, **k):
        pass

    def get_document_count(self, *a, **k):
        pass

    def get_documents_by_id(self, *a, **k):
        pass

    def get_label_count(self, *a, **k):
        pass

    def query_by_embedding(self, *a, **k):
        pass

    def write_documents(self, *a, **k):
        pass

    def write_labels(self, *a, **k):
        pass

    def delete_index(self, *a, **k):
        pass

    def update_document_meta(self, *a, **kw):
        pass


class MockRetriever(BaseRetriever):
    outgoing_edges = 1

    def retrieve(self, query: str, top_k: int):
        pass

    def retrieve_batch(self, queries: List[str], top_k: int):
        pass


class MockDenseRetriever(MockRetriever):
    def __init__(self, document_store: BaseDocumentStore, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.document_store = document_store

    def embed_queries(self, texts):
        return [np.random.rand(self.embedding_dim)] * len(texts)

    def embed_documents(self, docs):
        return [np.random.rand(self.embedding_dim)] * len(docs)


class MockReader(BaseReader):
    outgoing_edges = 1

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        pass

    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None, batch_size: Optional[int] = None):
        pass


#
# Document collections
#


@pytest.fixture
def docs_all_formats() -> List[Union[Document, Dict[str, Any]]]:
    return [
        # metafield at the top level for backward compatibility
        {
            "content": "My name is Paul and I live in New York",
            "meta_field": "test2",
            "name": "filename2",
            "date_field": "2019-10-01",
            "numeric_field": 5.0,
        },
        # "dict" format
        {
            "content": "My name is Carla and I live in Berlin",
            "meta": {"meta_field": "test1", "name": "filename1", "date_field": "2020-03-01", "numeric_field": 5.5},
        },
        # Document object
        Document(
            content="My name is Christelle and I live in Paris",
            meta={"meta_field": "test3", "name": "filename3", "date_field": "2018-10-01", "numeric_field": 4.5},
        ),
        Document(
            content="My name is Camila and I live in Madrid",
            meta={"meta_field": "test4", "name": "filename4", "date_field": "2021-02-01", "numeric_field": 3.0},
        ),
        Document(
            content="My name is Matteo and I live in Rome",
            meta={"meta_field": "test5", "name": "filename5", "date_field": "2019-01-01", "numeric_field": 0.0},
        ),
    ]


@pytest.fixture
def docs(docs_all_formats) -> List[Document]:
    return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]


@pytest.fixture
def docs_with_ids(docs) -> List[Document]:
    # Should be already sorted
    uuids = [
        UUID("190a2421-7e48-4a49-a639-35a86e202dfb"),
        UUID("20ff1706-cb55-4704-8ae8-a3459774c8dc"),
        UUID("5078722f-07ae-412d-8ccb-b77224c4bacb"),
        UUID("81d8ca45-fad1-4d1c-8028-d818ef33d755"),
        UUID("f985789f-1673-4d8f-8d5f-2b8d3a9e8e23"),
    ]
    uuids.sort()
    for doc, uuid in zip(docs, uuids):
        doc.id = str(uuid)
    return docs


@pytest.fixture
def docs_with_random_emb(docs) -> List[Document]:
    for doc in docs:
        doc.embedding = np.random.random([768])
    return docs


@pytest.fixture
def docs_with_true_emb():
    return [
        Document(
            content="The capital of Germany is the city state of Berlin.",
            embedding=np.loadtxt(SAMPLES_PATH / "embeddings" / "embedding_1.txt"),
        ),
        Document(
            content="Berlin is the capital and largest city of Germany by both area and population.",
            embedding=np.loadtxt(SAMPLES_PATH / "embeddings" / "embedding_2.txt"),
        ),
    ]


@pytest.fixture(autouse=True)
def gc_cleanup(request):
    """
    Run garbage collector between tests in order to reduce memory footprint for CI.
    """
    yield
    gc.collect()


@pytest.fixture(scope="session")
def elasticsearch_fixture():
    # test if a ES cluster is already running. If not, download and start an ES instance locally.
    try:
        client = Elasticsearch(hosts=[{"host": "localhost", "port": "9200"}])
        client.info()
    except:
        print("Starting Elasticsearch ...")
        status = subprocess.run(["docker rm haystack_test_elastic"], shell=True)
        status = subprocess.run(
            [
                'docker run -d --name haystack_test_elastic -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'
            ],
            shell=True,
        )
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch. Please check docker container logs.")
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
        status = subprocess.run(
            [
                "docker run -d --name milvus_cpu_0.10.5 -p 19530:19530 -p 19121:19121 "
                "milvusdb/milvus:0.10.5-cpu-d010621-4eda95"
            ],
            shell=True,
        )
        time.sleep(40)


@pytest.fixture(scope="session")
def weaviate_fixture():
    # test if a Weaviate server is already running. If not, start Weaviate docker container locally.
    # Make sure you have given > 6GB memory to docker engine
    try:
        weaviate_server = weaviate.Client(url="http://localhost:8080", timeout_config=(5, 15))
        weaviate_server.is_ready()
    except:
        print("Starting Weaviate servers ...")
        status = subprocess.run(["docker rm haystack_test_weaviate"], shell=True)
        status = subprocess.run(
            ["docker run -d --name haystack_test_weaviate -p 8080:8080 semitechnologies/weaviate:1.11.0"], shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch Weaviate. Please check docker container logs.")
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
        status = subprocess.run(["docker rm haystack_test_graphdb"], shell=True)
        status = subprocess.run(
            [
                "docker run -d -p 7200:7200 --name haystack_test_graphdb docker-registry.ontotext.com/graphdb-free:9.4.1-adoptopenjdk11"
            ],
            shell=True,
        )
        if status.returncode:
            raise Exception("Failed to launch GraphDB. Please check docker container logs.")
        time.sleep(30)


@pytest.fixture(scope="session")
def tika_fixture():
    try:
        tika_url = "http://localhost:9998/tika"
        ping = requests.get(tika_url)
        if ping.status_code != 200:
            raise Exception("Unable to connect Tika. Please check tika endpoint {0}.".format(tika_url))
    except:
        print("Starting Tika ...")
        status = subprocess.run(["docker run -d --name tika -p 9998:9998 apache/tika:1.24.1"], shell=True)
        if status.returncode:
            raise Exception("Failed to launch Tika. Please check docker container logs.")
        time.sleep(30)


@pytest.fixture(scope="session")
def xpdf_fixture():
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
                       {1} cp xpdf-tools-{0}-4.03/bin64/pdftotext /usr/local/bin""".format(
            platform_id, sudo_prefix
        )
        run([commands], shell=True)

        verify_installation = run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                 You can download for your OS from here: https://www.xpdfreader.com/download.html."""
            )


@pytest.fixture
def deepset_cloud_fixture():
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}",
            match=[responses.matchers.header_matcher({"authorization": f"Bearer {DC_API_KEY}"})],
            json={"indexing": {"status": "INDEXED", "pending_file_count": 0, "total_file_count": 31}},
            status=200,
        )
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines",
            match=[responses.matchers.header_matcher({"authorization": f"Bearer {DC_API_KEY}"})],
            json={
                "data": [
                    {
                        "name": DC_TEST_INDEX,
                        "status": "DEPLOYED",
                        "indexing": {"status": "INDEXED", "pending_file_count": 0, "total_file_count": 31},
                    }
                ],
                "has_more": False,
                "total": 1,
            },
        )
    else:
        responses.add_passthru(DC_API_ENDPOINT)


@pytest.fixture
@responses.activate
def deepset_cloud_document_store(deepset_cloud_fixture):
    return DeepsetCloudDocumentStore(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, index=DC_TEST_INDEX)


@pytest.fixture
def rag_generator():
    return RAGenerator(model_name_or_path="facebook/rag-token-nq", generator_type="token", max_length=20)


@pytest.fixture
def openai_generator():
    return OpenAIAnswerGenerator(api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1)


@pytest.fixture
def question_generator():
    return QuestionGenerator(model_name_or_path="valhalla/t5-small-e2e-qg")


@pytest.fixture
def lfqa_generator(request):
    return Seq2SeqGenerator(model_name_or_path=request.param, min_length=100, max_length=200)


@pytest.fixture
def summarizer():
    return TransformersSummarizer(model_name_or_path="google/pegasus-xsum", use_gpu=-1)


@pytest.fixture
def en_to_de_translator():
    return TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")


@pytest.fixture
def de_to_en_translator():
    return TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")


@pytest.fixture
def reader_without_normalized_scores():
    return FARMReader(
        model_name_or_path="distilbert-base-uncased-distilled-squad",
        use_gpu=False,
        top_k_per_sample=5,
        num_processes=0,
        use_confidence_scores=False,
    )


@pytest.fixture(params=["farm", "transformers"])
def reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="distilbert-base-uncased-distilled-squad",
            use_gpu=False,
            top_k_per_sample=5,
            num_processes=0,
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="distilbert-base-uncased-distilled-squad",
            tokenizer="distilbert-base-uncased",
            use_gpu=-1,
        )


@pytest.fixture(params=["tapas", "rci"])
def table_reader(request):
    if request.param == "tapas":
        return TableReader(model_name_or_path="google/tapas-base-finetuned-wtq")
    elif request.param == "rci":
        return RCIReader(
            row_model_name_or_path="michaelrglass/albert-base-rci-wikisql-row",
            column_model_name_or_path="michaelrglass/albert-base-rci-wikisql-col",
        )


@pytest.fixture
def ranker_two_logits():
    return SentenceTransformersRanker(model_name_or_path="deepset/gbert-base-germandpr-reranking")


@pytest.fixture
def ranker():
    return SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")


@pytest.fixture
def document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion", use_gpu=False
    )


@pytest.fixture
def zero_shot_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="cross-encoder/nli-distilroberta-base",
        use_gpu=False,
        task="zero-shot-classification",
        labels=["negative", "positive"],
    )


@pytest.fixture
def batched_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion", use_gpu=False, batch_size=16
    )


@pytest.fixture
def indexing_document_classifier():
    return TransformersDocumentClassifier(
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion",
        use_gpu=False,
        batch_size=16,
        classification_field="class_field",
    )


# TODO Fix bug in test_no_answer_output when using
# @pytest.fixture(params=["farm", "transformers"])
@pytest.fixture(params=["farm"])
def no_answer_reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="deepset/roberta-base-squad2",
            use_gpu=False,
            top_k_per_sample=5,
            no_ans_boost=0,
            return_no_answer=True,
            num_processes=0,
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            use_gpu=-1,
            top_k_per_candidate=5,
        )


@pytest.fixture
def prediction(reader, docs):
    prediction = reader.predict(query="Who lives in Berlin?", documents=docs, top_k=5)
    return prediction


@pytest.fixture
def no_answer_prediction(no_answer_reader, docs):
    prediction = no_answer_reader.predict(query="What is the meaning of life?", documents=docs, top_k=5)
    return prediction


@pytest.fixture(params=["es_filter_only", "elasticsearch", "dpr", "embedding", "tfidf", "table_text_retriever"])
def retriever(request, document_store):
    return get_retriever(request.param, document_store)


# @pytest.fixture(params=["es_filter_only", "elasticsearch", "dpr", "embedding", "tfidf"])
@pytest.fixture(params=["tfidf"])
def retriever_with_docs(request, document_store_with_docs):
    return get_retriever(request.param, document_store_with_docs)


def get_retriever(retriever_type, document_store):

    if retriever_type == "dpr":
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=False,
            embed_title=True,
        )
    elif retriever_type == "mdr":
        retriever = MultihopEmbeddingRetriever(
            document_store=document_store,
            embedding_model="deutschmann/mdr_roberta_q_encoder",  # or "facebook/dpr-ctx_encoder-single-nq-base"
            use_gpu=False,
        )
    elif retriever_type == "tfidf":
        retriever = TfidfRetriever(document_store=document_store)
        retriever.fit()
    elif retriever_type == "embedding":
        retriever = EmbeddingRetriever(
            document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False
        )
    elif retriever_type == "embedding_sbert":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/msmarco-distilbert-base-tas-b",
            model_format="sentence_transformers",
            use_gpu=False,
        )
    elif retriever_type == "retribert":
        retriever = EmbeddingRetriever(
            document_store=document_store, embedding_model="yjernite/retribert-base-uncased", use_gpu=False
        )
    elif retriever_type == "dpr_lfqa":
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
            passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
            use_gpu=False,
            embed_title=True,
        )
    elif retriever_type == "elasticsearch":
        retriever = BM25Retriever(document_store=document_store)
    elif retriever_type == "es_filter_only":
        retriever = FilterRetriever(document_store=document_store)
    elif retriever_type == "table_text_retriever":
        retriever = TableTextRetriever(
            document_store=document_store,
            query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
            passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
            table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
            use_gpu=False,
        )
    else:
        raise Exception(f"No retriever fixture for '{retriever_type}'")

    return retriever


def ensure_ids_are_correct_uuids(docs: list, document_store: object) -> None:
    # Weaviate currently only supports UUIDs
    if type(document_store) == WeaviateDocumentStore:
        for d in docs:
            d["id"] = str(uuid.uuid4())


# FIXME Fix this in the docstore tests refactoring
from inspect import getmembers, isclass, isfunction


def mock_pinecone(monkeypatch):
    for fname, function in getmembers(pinecone_mock, isfunction):
        monkeypatch.setattr(f"pinecone.{fname}", function, raising=False)
    for cname, class_ in getmembers(pinecone_mock, isclass):
        monkeypatch.setattr(f"pinecone.{cname}", class_, raising=False)


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate", "pinecone"])
def document_store_with_docs(request, docs, tmp_path, monkeypatch):
    if request.param == "pinecone":
        mock_pinecone(monkeypatch)

    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(768))
    document_store = get_document_store(
        document_store_type=request.param, embedding_dim=embedding_dim.args[0], tmp_path=tmp_path
    )
    document_store.write_documents(docs)
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture
def document_store(request, tmp_path, monkeypatch: pytest.MonkeyPatch):
    if request.param == "pinecone":
        mock_pinecone(monkeypatch)

    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(768))
    document_store = get_document_store(
        document_store_type=request.param, embedding_dim=embedding_dim.args[0], tmp_path=tmp_path
    )
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture(params=["memory", "faiss", "milvus1", "milvus", "elasticsearch", "pinecone"])
def document_store_dot_product(request, tmp_path, monkeypatch):
    if request.param == "pinecone":
        mock_pinecone(monkeypatch)

    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(768))
    document_store = get_document_store(
        document_store_type=request.param,
        embedding_dim=embedding_dim.args[0],
        similarity="dot_product",
        tmp_path=tmp_path,
    )
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture(params=["memory", "faiss", "milvus1", "milvus", "elasticsearch", "pinecone"])
def document_store_dot_product_with_docs(request, docs, tmp_path, monkeypatch):
    if request.param == "pinecone":
        mock_pinecone(monkeypatch)

    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(768))
    document_store = get_document_store(
        document_store_type=request.param,
        embedding_dim=embedding_dim.args[0],
        similarity="dot_product",
        tmp_path=tmp_path,
    )
    document_store.write_documents(docs)
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "milvus1", "pinecone"])
def document_store_dot_product_small(request, tmp_path, monkeypatch):
    if request.param == "pinecone":
        mock_pinecone(monkeypatch)

    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(3))
    document_store = get_document_store(
        document_store_type=request.param,
        embedding_dim=embedding_dim.args[0],
        similarity="dot_product",
        tmp_path=tmp_path,
    )
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate", "pinecone"])
def document_store_small(request, tmp_path, monkeypatch):
    if request.param == "pinecone":
        mock_pinecone(monkeypatch)

    embedding_dim = request.node.get_closest_marker("embedding_dim", pytest.mark.embedding_dim(3))
    document_store = get_document_store(
        document_store_type=request.param, embedding_dim=embedding_dim.args[0], similarity="cosine", tmp_path=tmp_path
    )
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture(autouse=True)
def postgres_fixture():
    if SQL_TYPE == "postgres":
        setup_postgres()
        yield
        teardown_postgres()
    else:
        yield


@pytest.fixture
def sql_url(tmp_path):
    return get_sql_url(tmp_path)


def get_sql_url(tmp_path):
    if SQL_TYPE == "postgres":
        return "postgresql://postgres:postgres@127.0.0.1/postgres"
    else:
        return f"sqlite:///{tmp_path}/haystack_test.db"


def setup_postgres():
    # status = subprocess.run(["docker run --name postgres_test -d -e POSTGRES_HOST_AUTH_METHOD=trust -p 5432:5432 postgres"], shell=True)
    # if status.returncode:
    #     logging.warning("Tried to start PostgreSQL through Docker but this failed. It is likely that there is already an existing instance running.")
    # else:
    #     sleep(5)
    engine = create_engine("postgresql://postgres:postgres@127.0.0.1/postgres", isolation_level="AUTOCOMMIT")

    with engine.connect() as connection:
        try:
            connection.execute(text("DROP SCHEMA public CASCADE"))
        except Exception as e:
            logging.error(e)
        connection.execute(text("CREATE SCHEMA public;"))
        connection.execute(text('SET SESSION idle_in_transaction_session_timeout = "1s";'))


def teardown_postgres():
    engine = create_engine("postgresql://postgres:postgres@127.0.0.1/postgres", isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.close()


def get_document_store(
    document_store_type,
    tmp_path,
    embedding_dim=768,
    embedding_field="embedding",
    index="haystack_test",
    similarity: str = "cosine",
):  # cosine is default similarity as dot product is not supported by Weaviate
    if document_store_type == "sql":
        document_store = SQLDocumentStore(url=get_sql_url(tmp_path), index=index, isolation_level="AUTOCOMMIT")

    elif document_store_type == "memory":
        document_store = InMemoryDocumentStore(
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
        )

    elif document_store_type == "elasticsearch":
        # make sure we start from a fresh index
        document_store = ElasticsearchDocumentStore(
            index=index,
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            similarity=similarity,
            recreate_index=True,
        )

    elif document_store_type == "faiss":
        document_store = FAISSDocumentStore(
            embedding_dim=embedding_dim,
            sql_url=get_sql_url(tmp_path),
            return_embedding=True,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            isolation_level="AUTOCOMMIT",
        )

    elif document_store_type == "milvus1":
        document_store = MilvusDocumentStore(
            embedding_dim=embedding_dim,
            sql_url=get_sql_url(tmp_path),
            return_embedding=True,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            isolation_level="AUTOCOMMIT",
        )

    elif document_store_type == "milvus":
        document_store = MilvusDocumentStore(
            embedding_dim=embedding_dim,
            sql_url=get_sql_url(tmp_path),
            return_embedding=True,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            isolation_level="AUTOCOMMIT",
            recreate_index=True,
        )

    elif document_store_type == "weaviate":
        document_store = WeaviateDocumentStore(
            index=index, similarity=similarity, embedding_dim=embedding_dim, recreate_index=True
        )

    elif document_store_type == "pinecone":
        document_store = PineconeDocumentStore(
            api_key=os.environ.get("PINECONE_API_KEY"),
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            recreate_index=True,
        )

    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")

    return document_store


@pytest.fixture
def adaptive_model_qa(num_processes):
    """
    PyTest Fixture for a Question Answering Inferencer based on PyTorch.
    """
    try:
        model = Inferencer.load(
            "deepset/bert-base-cased-squad2",
            task_type="question_answering",
            batch_size=16,
            num_processes=num_processes,
            gpu=False,
        )
        yield model
    finally:
        if num_processes != 0:
            # close the pool
            # we pass join=True to wait for all sub processes to close
            # this is because below we want to test if all sub-processes
            # have exited
            model.close_multiprocessing_pool(join=True)

    # check if all workers (sub processes) are closed
    current_process = psutil.Process()
    children = current_process.children()
    if len(children) != 0:
        logging.error(f"Not all the subprocesses are closed! {len(children)} are still running.")


@pytest.fixture
def bert_base_squad2(request):
    model = QAInferencer.load(
        "deepset/minilm-uncased-squad2",
        task_type="question_answering",
        batch_size=4,
        num_processes=0,
        multithreading_rust=False,
        use_fast=True,  # TODO parametrize this to test slow as well
    )
    return model
