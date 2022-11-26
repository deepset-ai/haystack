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
import re

import requests_cache
import responses
from sqlalchemy import create_engine, text
import posthog

import numpy as np
import psutil
import pytest
import requests

from haystack import Answer, BaseComponent
from haystack.document_stores import (
    BaseDocumentStore,
    InMemoryDocumentStore,
    ElasticsearchDocumentStore,
    WeaviateDocumentStore,
    MilvusDocumentStore,
    PineconeDocumentStore,
    OpenSearchDocumentStore,
    GraphDBKnowledgeGraph,
    FAISSDocumentStore,
    SQLDocumentStore,
)
from haystack.nodes import (
    BaseReader,
    BaseRetriever,
    OpenAIAnswerGenerator,
    BaseGenerator,
    BaseSummarizer,
    BaseTranslator,
    DenseRetriever,
    Seq2SeqGenerator,
    RAGenerator,
    SentenceTransformersRanker,
    TransformersDocumentClassifier,
    FilterRetriever,
    BM25Retriever,
    TfidfRetriever,
    DensePassageRetriever,
    EmbeddingRetriever,
    MultihopEmbeddingRetriever,
    TableTextRetriever,
    FARMReader,
    TransformersReader,
    TableReader,
    RCIReader,
    TransformersSummarizer,
    TransformersTranslator,
    QuestionGenerator,
)
from haystack.modeling.infer import Inferencer, QAInferencer
from haystack.schema import Document
from haystack.utils.import_utils import _optional_component_not_installed

try:
    from elasticsearch import Elasticsearch
    import weaviate
except (ImportError, ModuleNotFoundError) as ie:
    _optional_component_not_installed("test", "test", ie)

from .mocks import pinecone as pinecone_mock


# To manually run the tests with default PostgreSQL instead of SQLite, switch the lines below
SQL_TYPE = "sqlite"
SAMPLES_PATH = Path(__file__).parent / "samples"
DC_API_ENDPOINT = "https://DC_API/v1"
DC_TEST_INDEX = "document_retrieval_1"
DC_API_KEY = "NO_KEY"
MOCK_DC = True

# Set metadata fields used during testing for PineconeDocumentStore meta_config
META_FIELDS = [
    "meta_field",
    "name",
    "date_field",
    "numeric_field",
    "f1",
    "f3",
    "meta_id",
    "meta_field_for_count",
    "meta_key_1",
    "meta_key_2",
]

# Disable telemetry reports when running tests
posthog.disabled = True

# Cache requests (e.g. huggingface model) to circumvent load protection
# See https://requests-cache.readthedocs.io/en/stable/user_guide/filtering.html
requests_cache.install_cache(urls_expire_after={"huggingface.co": timedelta(hours=1), "*": requests_cache.DO_NOT_CACHE})


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
        "milvus": [pytest.mark.milvus],
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

        required_doc_store = infer_required_doc_store(item, keywords)

        if required_doc_store and required_doc_store not in document_store_types_to_run:
            skip_docstore = pytest.mark.skip(
                reason=f'{required_doc_store} is disabled. Enable via pytest --document_store_type="{required_doc_store}"'
            )
            item.add_marker(skip_docstore)


def infer_required_doc_store(item, keywords):
    # assumption: a test runs only with one document_store
    # if there are multiple docstore markers, we apply the following heuristics:
    # 1. if the test was parameterized, we use the the parameter
    # 2. if the test name contains the docstore name, we use that
    # 3. use an arbitrary one by calling set.pop()
    required_doc_store = None
    all_doc_stores = {"elasticsearch", "faiss", "sql", "memory", "milvus", "weaviate", "pinecone"}
    docstore_markers = set(keywords).intersection(all_doc_stores)
    if len(docstore_markers) > 1:
        # if parameterized infer the docstore from the parameter
        if hasattr(item, "callspec"):
            for doc_store in all_doc_stores:
                # callspec.id contains the parameter values of the test
                if re.search(f"(^|-){doc_store}($|[-_])", item.callspec.id):
                    required_doc_store = doc_store
                    break
        # if still not found, infer the docstore from the test name
        if required_doc_store is None:
            for doc_store in all_doc_stores:
                if doc_store in item.name:
                    required_doc_store = doc_store
                    break
    # if still not found or there is only one, use an arbitrary one from the markers
    if required_doc_store is None:
        required_doc_store = docstore_markers.pop() if docstore_markers else None
    return required_doc_store


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


class MockSeq2SegGenerator(BaseGenerator):
    def predict(self, query: str, documents: List[Document], top_k: Optional[int]) -> Dict:
        pass


class MockSummarizer(BaseSummarizer):
    def predict_batch(
        self,
        documents: Union[List[Document], List[List[Document]]],
        generate_single_summary: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def predict(self, documents: List[Document], generate_single_summary: Optional[bool] = None) -> List[Document]:
        pass


class MockTranslator(BaseTranslator):
    def translate(
        self,
        results: List[Dict[str, Any]] = None,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
    ) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]:
        pass

    def translate_batch(
        self,
        queries: Optional[List[str]] = None,
        documents: Optional[Union[List[Document], List[Answer], List[List[Document]], List[List[Answer]]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]]:
        pass


class MockDenseRetriever(MockRetriever, DenseRetriever):
    def __init__(self, document_store: BaseDocumentStore, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.document_store = document_store

    def embed_queries(self, queries):
        return np.random.rand(len(queries), self.embedding_dim)

    def embed_documents(self, documents):
        return np.random.rand(len(documents), self.embedding_dim)


class MockQuestionGenerator(QuestionGenerator):
    def __init__(self):
        pass

    def predict(self, query: str, documents: List[Document], top_k: Optional[int]) -> Dict:
        pass


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
            ["docker run -d --name haystack_test_weaviate -p 8080:8080 semitechnologies/weaviate:latest"], shell=True
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
        status = subprocess.run(["docker run -d --name tika -p 9998:9998 apache/tika:1.28.4"], shell=True)
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
    return TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-6", use_gpu=False)


@pytest.fixture
def en_to_de_translator():
    return TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")


@pytest.fixture
def de_to_en_translator():
    return TransformersTranslator(model_name_or_path="Helsinki-NLP/opus-mt-de-en")


@pytest.fixture
def reader_without_normalized_scores():
    return FARMReader(
        model_name_or_path="deepset/bert-medium-squad2-distilled",
        use_gpu=False,
        top_k_per_sample=5,
        num_processes=0,
        use_confidence_scores=False,
    )


@pytest.fixture(params=["farm", "transformers"], scope="module")
def reader(request):
    if request.param == "farm":
        return FARMReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            use_gpu=False,
            top_k_per_sample=5,
            num_processes=0,
        )
    if request.param == "transformers":
        return TransformersReader(
            model_name_or_path="deepset/bert-medium-squad2-distilled",
            tokenizer="deepset/bert-medium-squad2-distilled",
            use_gpu=-1,
        )


@pytest.fixture(params=["tapas_small", "tapas_base", "tapas_scored", "rci"])
def table_reader_and_param(request):
    if request.param == "tapas_small":
        return TableReader(model_name_or_path="google/tapas-small-finetuned-wtq"), request.param
    elif request.param == "tapas_base":
        return TableReader(model_name_or_path="google/tapas-base-finetuned-wtq"), request.param
    elif request.param == "tapas_scored":
        return TableReader(model_name_or_path="deepset/tapas-large-nq-hn-reader"), request.param
    elif request.param == "rci":
        return (
            RCIReader(
                row_model_name_or_path="michaelrglass/albert-base-rci-wikisql-row",
                column_model_name_or_path="michaelrglass/albert-base-rci-wikisql-col",
            ),
            request.param,
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
        model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion", use_gpu=False, top_k=2
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


@pytest.fixture(params=["es_filter_only", "bm25", "dpr", "embedding", "tfidf", "table_text_retriever"])
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
    elif retriever_type == "openai":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="ada",
            use_gpu=False,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    elif retriever_type == "cohere":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="small",
            use_gpu=False,
            api_key=os.environ.get("COHERE_API_KEY", ""),
        )
    elif retriever_type == "dpr_lfqa":
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
            passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
            use_gpu=False,
            embed_title=True,
        )
    elif retriever_type == "bm25":
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


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "milvus", "weaviate", "pinecone"])
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


@pytest.fixture(params=["memory", "faiss", "milvus", "elasticsearch", "pinecone"])
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


@pytest.fixture(params=["memory", "faiss", "milvus", "elasticsearch", "pinecone", "weaviate"])
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


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "milvus", "pinecone"])
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


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "milvus", "weaviate", "pinecone"])
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
            connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
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
    recreate_index: bool = True,
):  # cosine is default similarity as dot product is not supported by Weaviate
    document_store: BaseDocumentStore
    if document_store_type == "memory":
        document_store = InMemoryDocumentStore(
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            use_bm25=True,
        )

    elif document_store_type == "elasticsearch":
        # make sure we start from a fresh index
        document_store = ElasticsearchDocumentStore(
            index=index,
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            similarity=similarity,
            recreate_index=recreate_index,
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

    elif document_store_type == "milvus":
        document_store = MilvusDocumentStore(
            embedding_dim=embedding_dim,
            sql_url=get_sql_url(tmp_path),
            return_embedding=True,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            isolation_level="AUTOCOMMIT",
            recreate_index=recreate_index,
        )

    elif document_store_type == "weaviate":
        document_store = WeaviateDocumentStore(
            index=index, similarity=similarity, embedding_dim=embedding_dim, recreate_index=recreate_index
        )

    elif document_store_type == "pinecone":
        document_store = PineconeDocumentStore(
            api_key=os.environ.get("PINECONE_API_KEY") or "fake-haystack-test-key",
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            index=index,
            similarity=similarity,
            recreate_index=recreate_index,
            metadata_config={"indexed": META_FIELDS},
        )

    elif document_store_type == "opensearch_faiss":
        document_store = OpenSearchDocumentStore(
            index=index,
            return_embedding=True,
            embedding_dim=embedding_dim,
            embedding_field=embedding_field,
            similarity=similarity,
            recreate_index=recreate_index,
            port=9201,
            knn_engine="faiss",
        )

    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")

    return document_store


@pytest.fixture
def adaptive_model_qa(num_processes):
    """
    PyTest Fixture for a Question Answering Inferencer based on PyTorch.
    """

    model = Inferencer.load(
        "deepset/bert-medium-squad2-distilled",
        task_type="question_answering",
        batch_size=16,
        num_processes=num_processes,
        gpu=False,
    )
    yield model

    # check if all workers (sub processes) are closed
    current_process = psutil.Process()
    children = current_process.children()
    if len(children) != 0:
        logging.error("Not all the subprocesses are closed! %s are still running.", len(children))


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
