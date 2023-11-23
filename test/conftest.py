import warnings
from typing import Any, List, Optional, Dict, Union

import gc
import logging
from pathlib import Path
import os
import re
from functools import wraps
from unittest.mock import patch

import responses
import posthog

import numpy as np
import pytest

from haystack import Answer, BaseComponent, __version__ as haystack_version
from haystack.document_stores import (
    BaseDocumentStore,
    InMemoryDocumentStore,
    ElasticsearchDocumentStore,
    WeaviateDocumentStore,
    PineconeDocumentStore,
    OpenSearchDocumentStore,
    FAISSDocumentStore,
)
from haystack.nodes import (
    BaseReader,
    BaseRetriever,
    BaseGenerator,
    BaseSummarizer,
    BaseTranslator,
    DenseRetriever,
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
    QuestionGenerator,
    PromptTemplate,
)
from haystack.nodes.prompt import PromptNode
from haystack.schema import Document, FilterType, MultiLabel, Label, Span

from .mocks import pinecone as pinecone_mock


# To manually run the tests with default PostgreSQL instead of SQLite, switch the lines below
SQL_TYPE = "sqlite"
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

# Disable caching from prompthub to avoid polluting the local environment.
os.environ["PROMPTHUB_CACHE_ENABLED"] = "false"


def fail_at_version(target_major, target_minor):
    """
    Reminder to remove deprecated features.
    If you're using this fixture please open an issue in the repo to keep track
    of the deprecated feature that must be removed.
    After opening the issue assign it to the target version milestone, if the
    milestone doesn't exist either create it or notify someone that has permissions
    to do so.
    This way will be assured that the feature is actually removed for that release.
    This will fail tests if the current major and/or minor version is equal or greater
    of target_major and/or target_minor.
    If the current version has `rc0` set the test won't fail but only issue a warning, this
    is done because we use `rc0` to mark the development version in `main`. If we wouldn't
    do this tests would continuously fail in main.

    ```python
    from ..conftest import fail_at_version

    @fail_at_version(1, 10)  # Will fail once Haystack version is greater than or equal to 1.10
    def test_test():
        assert True
    ```
    """

    def decorator(function):
        (current_major, current_minor) = [int(num) for num in haystack_version.split(".")[:2]]
        current_rc = int(haystack_version.split("rc")[1]) if "rc" in haystack_version else -1

        @wraps(function)
        def wrapper(*args, **kwargs):
            if current_major > target_major or (current_major == target_major and current_minor >= target_minor):
                message = f"This feature is marked for removal in v{target_major}.{target_minor}"
                if current_rc == 0:
                    warnings.warn(message)
                else:
                    pytest.fail(reason=message)
            return_value = function(*args, **kwargs)
            return return_value

        return wrapper

    return decorator


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
        "weaviate": [pytest.mark.weaviate],
        "pinecone": [pytest.mark.pinecone],
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
    all_doc_stores = {"elasticsearch", "faiss", "sql", "memory", "weaviate", "pinecone"}
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

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        return []

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        return [[]]


class MockBaseRetriever(MockRetriever):
    def __init__(self, document_store: BaseDocumentStore, mock_document: Document):
        self.document_store = document_store
        self.mock_document = mock_document

    def retrieve(
        self,
        query: str,
        filters: dict,
        top_k: Optional[int],
        index: str,
        headers: Optional[Dict[str, str]],
        scale_score: bool,
    ):
        return [self.mock_document]

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
    ):
        return [[self.mock_document] for _ in range(len(queries))]

    def embed_documents(self, documents: List[Document]):
        return np.full((len(documents), 768), 0.5)


class MockSeq2SegGenerator(BaseGenerator):
    def predict(self, query: str, documents: List[Document], top_k: Optional[int], max_tokens: Optional[int]) -> Dict:
        pass


class MockSummarizer(BaseSummarizer):
    def predict_batch(
        self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def predict(self, documents: List[Document]) -> List[Document]:
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


class MockPromptNode(PromptNode):
    def __init__(self):
        self.default_prompt_template = None
        self.model_name_or_path = ""

    def prompt(self, prompt_template: Optional[Union[str, PromptTemplate]], *args, **kwargs) -> List[str]:
        return [""]

    def get_prompt_template(self, prompt_template: Union[str, PromptTemplate, None]) -> Optional[PromptTemplate]:
        if prompt_template == "think-step-by-step":
            p = PromptTemplate(
                "You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions "
                "correctly, you have access to the following tools:\n\n"
                "{tool_names_with_descriptions}\n\n"
                "To answer questions, you'll need to go through multiple steps involving step-by-step thinking and "
                "selecting appropriate tools and their inputs; tools will respond with observations. When you are ready "
                "for a final answer, respond with the `Final Answer:`\n\n"
                "Use the following format:\n\n"
                "Question: the question to be answered\n"
                "Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.\n"
                "Tool: [{tool_names}]\n"
                "Tool Input: the input for the tool\n"
                "Observation: the tool will respond with the result\n"
                "...\n"
                "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
                "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass\n"
                "---\n\n"
                "Question: {query}\n"
                "Thought: Let's think step-by-step, I first need to {generated_text}"
            )
            p.name = "think-step-by-step"
        else:
            return PromptTemplate("test prompt")


@pytest.fixture
def test_rootdir() -> Path:
    return Path(__file__).parent.absolute()


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
            "list_field": ["item0.1", "item0.2"],
        },
        # "dict" format
        {
            "content": "My name is Carla and I live in Berlin",
            "meta": {
                "meta_field": "test1",
                "name": "filename1",
                "date_field": "2020-03-01",
                "numeric_field": 5.5,
                "list_field": ["item1.1", "item1.2"],
            },
        },
        # Document object
        Document(
            content="My name is Christelle and I live in Paris",
            meta={
                "meta_field": "test3",
                "name": "filename3",
                "date_field": "2018-10-01",
                "numeric_field": 4.5,
                "list_field": ["item2.1", "item2.2"],
            },
        ),
        Document(
            content="My name is Camila and I live in Madrid",
            meta={
                "meta_field": "test4",
                "name": "filename4",
                "date_field": "2021-02-01",
                "numeric_field": 3.0,
                "list_field": ["item3.1", "item3.2"],
            },
        ),
        Document(
            content="My name is Matteo and I live in Rome",
            meta={
                "meta_field": "test5",
                "name": "filename5",
                "date_field": "2019-01-01",
                "numeric_field": 0.0,
                "list_field": ["item4.1", "item4.2"],
            },
        ),
    ]


@pytest.fixture
def docs(docs_all_formats) -> List[Document]:
    return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]


@pytest.fixture(autouse=True)
def gc_cleanup(request):
    """
    Run garbage collector between tests in order to reduce memory footprint for CI.
    """
    yield
    gc.collect()


@pytest.fixture
def eval_labels() -> List[MultiLabel]:
    EVAL_LABELS = [
        MultiLabel(
            labels=[
                Label(
                    query="Who lives in Berlin?",
                    answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="a0747b83aea0b60c4b114b15476dd32d",
                        content_type="text",
                        content="My name is Carla and I live in Berlin",
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        ),
        MultiLabel(
            labels=[
                Label(
                    query="Who lives in Munich?",
                    answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]),
                    document=Document(
                        id="something_else", content_type="text", content="My name is Carla and I live in Munich"
                    ),
                    is_correct_answer=True,
                    is_correct_document=True,
                    origin="gold-label",
                )
            ]
        ),
    ]
    return EVAL_LABELS


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
def question_generator():
    return QuestionGenerator(model_name_or_path="valhalla/t5-small-e2e-qg")


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
    elif retriever_type == "embedding_sbert_instructions":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/msmarco-distilbert-dot-v5",
            model_format="sentence_transformers",
            query_prompt="Embed this query for retrieval:",
            passage_prompt="Embed this passage for retrieval:",
            use_gpu=False,
        )
    elif retriever_type == "retribert":
        retriever = EmbeddingRetriever(
            document_store=document_store, embedding_model="yjernite/retribert-base-uncased", use_gpu=False
        )
    elif retriever_type == "openai":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="text-embedding-ada-002",
            use_gpu=False,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif retriever_type == "azure":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="text-embedding-ada-002",
            use_gpu=False,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
            azure_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED"),
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


# FIXME Fix this in the docstore tests refactoring
from inspect import getmembers, isclass, isfunction


def mock_pinecone(monkeypatch):
    for fname, function in getmembers(pinecone_mock, isfunction):
        monkeypatch.setattr(f"pinecone.{fname}", function, raising=False)
    for cname, class_ in getmembers(pinecone_mock, isclass):
        monkeypatch.setattr(f"pinecone.{cname}", class_, raising=False)


@pytest.fixture(params=["elasticsearch", "faiss", "memory", "weaviate", "pinecone"])
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


def get_sql_url(tmp_path):
    if SQL_TYPE == "postgres":
        return "postgresql://postgres:postgres@127.0.0.1/postgres"
    else:
        return f"sqlite:///{tmp_path}/haystack_test.db"


# TODO: Verify this is still necessary as it's called by no one
def setup_postgres():
    # status = subprocess.run(["docker run --name postgres_test -d -e POSTGRES_HOST_AUTH_METHOD=trust -p 5432:5432 postgres"], shell=True)
    # if status.returncode:
    #     logging.warning("Tried to start PostgreSQL through Docker but this failed. It is likely that there is already an existing instance running.")
    # else:
    #     sleep(5)
    from sqlalchemy import create_engine, text

    engine = create_engine("postgresql://postgres:postgres@127.0.0.1/postgres", isolation_level="AUTOCOMMIT")

    with engine.connect() as connection:
        try:
            connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        except Exception as e:
            logging.error(e)
        connection.execute(text("CREATE SCHEMA public;"))
        connection.execute(text('SET SESSION idle_in_transaction_session_timeout = "1s";'))


# TODO: Verify this is still necessary as it's called by no one
def teardown_postgres():
    from sqlalchemy import create_engine, text

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
            bm25_parameters={"k1": 1.2, "b": 0.75},  # parameters similar to those of Elasticsearch
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
def haystack_azure_conf():
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
    azure_base_url = os.environ.get("AZURE_OPENAI_BASE_URL", None)
    azure_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", None)
    if api_key and azure_base_url and azure_deployment_name:
        return {"api_key": api_key, "azure_base_url": azure_base_url, "azure_deployment_name": azure_deployment_name}
    else:
        return {}


@pytest.fixture
def haystack_openai_config(request, haystack_azure_conf):
    if request.param == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            return {}
        else:
            return {"api_key": api_key, "embedding_model": "text-embedding-ada-002"}
    elif request.param == "azure":
        return haystack_azure_conf

    return {}


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"


@pytest.fixture
def sample_txt_file_paths_list(samples_path):
    return list((samples_path / "docs").glob("*.txt"))


@pytest.fixture
def preview_samples_path():
    return Path(__file__).parent / "preview" / "test_files"


@pytest.fixture(autouse=True)
def request_blocker(request: pytest.FixtureRequest, monkeypatch):
    """
    This fixture is applied automatically to all tests.
    Those that are marked as unit will have the requests module
    monkeypatched to avoid making HTTP requests by mistake.
    """
    marker = request.node.get_closest_marker("unit")
    if marker is None:
        return

    def urlopen_mock(self, method, url, *args, **kwargs):
        raise RuntimeError(f"The test was about to {method} {self.scheme}://{self.host}{url}")

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen_mock)


@pytest.fixture
def mock_auto_tokenizer():
    with patch("transformers.AutoTokenizer.from_pretrained", autospec=True) as mock_from_pretrained:
        yield mock_from_pretrained
