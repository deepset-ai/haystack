import os
from unittest.mock import MagicMock, patch, Mock
from test.conftest import MockDocumentStore
import pytest

from haystack import Document, Pipeline
from haystack.document_stores.base import BaseDocumentStore
from haystack.nodes import WebRetriever, PromptNode
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.retriever.web import SearchResult
from test.nodes.conftest import example_serperdev_response


@pytest.fixture
def mocked_requests():
    with patch("haystack.nodes.retriever.link_content.requests") as mock_requests:
        mock_response = Mock()
        mock_requests.get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.text = "Sample content from webpage"
        yield mock_requests


@pytest.fixture
def mocked_article_extractor():
    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value="Sample content from webpage"):
        yield


@pytest.mark.unit
def test_init_default_parameters():
    retriever = WebRetriever(api_key="test_key")

    assert retriever.top_k == 5
    assert retriever.mode == "snippets"
    assert retriever.preprocessor is None
    assert retriever.cache_document_store is None
    assert retriever.cache_index is None
    assert retriever.cache_headers is None
    assert retriever.cache_time == 1 * 24 * 60 * 60


@pytest.mark.unit
def test_init_custom_parameters():
    preprocessor = PreProcessor()
    document_store = MagicMock(spec=BaseDocumentStore)
    headers = {"Test": "Header"}

    retriever = WebRetriever(
        api_key="test_key",
        search_engine_provider="SerperDev",
        top_search_results=15,
        top_k=7,
        mode="preprocessed_documents",
        preprocessor=preprocessor,
        cache_document_store=document_store,
        cache_index="custom_index",
        cache_headers=headers,
        cache_time=2 * 24 * 60 * 60,
    )

    assert retriever.top_k == 7
    assert retriever.mode == "preprocessed_documents"
    assert retriever.preprocessor == preprocessor
    assert retriever.cache_document_store == document_store
    assert retriever.cache_index == "custom_index"
    assert retriever.cache_headers == headers
    assert retriever.cache_time == 2 * 24 * 60 * 60


@pytest.mark.unit
def test_retrieve_from_web_all_params(mock_web_search):
    wr = WebRetriever(api_key="fake_key")

    preprocessor = PreProcessor()

    result = wr._retrieve_from_web(query_norm="who is the boyfriend of olivia wilde?", preprocessor=preprocessor)

    assert isinstance(result, list)
    assert all(isinstance(doc, Document) for doc in result)
    assert len(result) == len(example_serperdev_response["organic"])


@pytest.mark.unit
def test_retrieve_from_web_no_preprocessor(mock_web_search):
    # tests that we get top_k results when no PreProcessor is provided
    wr = WebRetriever(api_key="fake_key")
    result = wr._retrieve_from_web("query", None)

    assert isinstance(result, list)
    assert all(isinstance(doc, Document) for doc in result)
    assert len(result) == len(example_serperdev_response["organic"])


@pytest.mark.unit
def test_retrieve_from_web_invalid_query(mock_web_search):
    # however, if query is None or empty, we expect an error
    wr = WebRetriever(api_key="fake_key")
    with pytest.raises(ValueError, match="WebSearch run requires"):
        wr._retrieve_from_web("", None)

    with pytest.raises(ValueError, match="WebSearch run requires"):
        wr._retrieve_from_web(None, None)


@pytest.mark.unit
def test_prepare_links_empty_list():
    wr = WebRetriever(api_key="fake_key")
    result = wr._prepare_links([])
    assert result == []

    result = wr._prepare_links(None)
    assert result == []


@pytest.mark.unit
def test_scrape_links_empty_list():
    wr = WebRetriever(api_key="fake_key")
    result = wr._scrape_links([], "query", None)
    assert result == []


@pytest.mark.unit
def test_scrape_links_with_search_results(mocked_requests, mocked_article_extractor):
    wr = WebRetriever(api_key="fake_key")

    sr1 = SearchResult("https://pagesix.com", "Some text", "0.43", "1")
    sr2 = SearchResult("https://www.yahoo.com/", "Some text", "0.43", "2")
    fake_search_results = [sr1, sr2]

    result = wr._scrape_links(fake_search_results, "query", None)

    assert isinstance(result, list)
    assert all(isinstance(r, Document) for r in result)
    assert len(result) == 2


@pytest.mark.unit
def test_scrape_links_with_search_results_with_preprocessor(mocked_requests, mocked_article_extractor):
    wr = WebRetriever(api_key="fake_key", mode="preprocessed_documents")
    preprocessor = PreProcessor(progress_bar=False)

    sr1 = SearchResult("https://pagesix.com", "Some text", "0.43", "1")
    sr2 = SearchResult("https://www.yahoo.com/", "Some text", "0.43", "2")
    fake_search_results = [sr1, sr2]

    result = wr._scrape_links(fake_search_results, "query", preprocessor)

    assert isinstance(result, list)
    assert all(isinstance(r, Document) for r in result)
    # the documents from above SearchResult are so small that they will not be split into multiple documents
    # by the preprocessor
    assert len(result) == 2


@pytest.mark.unit
def test_retrieve_uses_defaults():
    wr = WebRetriever(api_key="fake_key")

    with patch.object(wr, "_check_cache", return_value=[]) as mock_check_cache:
        with patch.object(wr, "_retrieve_from_web", return_value=[]) as mock_retrieve_from_web:
            wr.retrieve("query")

    # cache is checked first, always
    mock_check_cache.assert_called_with(
        "query", cache_index=wr.cache_index, cache_headers=wr.cache_headers, cache_time=wr.cache_time
    )
    mock_retrieve_from_web.assert_called_with("query", wr.preprocessor)


@pytest.mark.unit
def test_retrieve_batch():
    queries = ["query1", "query2"]
    wr = WebRetriever(api_key="fake_key")
    web_docs = [Document("doc1"), Document("doc2"), Document("doc3")]
    with patch.object(wr, "_check_cache", return_value=[]) as mock_check_cache:
        with patch.object(wr, "_retrieve_from_web", return_value=web_docs) as mock_retrieve_from_web:
            result = wr.retrieve_batch(queries)

    assert mock_check_cache.call_count == len(queries)
    assert mock_retrieve_from_web.call_count == len(queries)
    # check that the result is a list of lists of Documents
    # where each list of Documents is the result of a single query
    assert len(result) == len(queries)

    # check that the result is a list of lists of Documents
    assert all(isinstance(docs, list) for docs in result)
    assert all(isinstance(doc, Document) for docs in result for doc in docs)

    # check that the result is a list of lists of Documents, so that the number of Documents
    # is equal to the number of queries * number of documents retrieved per query
    assert len([doc for docs in result for doc in docs]) == len(web_docs) * len(queries)


@pytest.mark.unit
def test_retrieve_uses_cache():
    wr = WebRetriever(api_key="fake_key")

    cached_docs = [Document("doc1"), Document("doc2")]
    with patch.object(wr, "_check_cache", return_value=cached_docs) as mock_check_cache:
        with patch.object(wr, "_retrieve_from_web") as mock_retrieve_from_web:
            with patch.object(wr, "_save_cache") as mock_save_cache:
                result = wr.retrieve("query")

    # checking cache is always called
    mock_check_cache.assert_called()

    # these methods are not called because we found docs in cache
    mock_retrieve_from_web.assert_not_called()
    mock_save_cache.assert_not_called()

    assert result == cached_docs


@pytest.mark.unit
def test_retrieve_saves_to_cache():
    wr = WebRetriever(api_key="fake_key", cache_document_store=MockDocumentStore())
    web_docs = [Document("doc1"), Document("doc2"), Document("doc3")]

    with patch.object(wr, "_check_cache", return_value=[]) as mock_check_cache:
        with patch.object(wr, "_retrieve_from_web", return_value=web_docs) as mock_retrieve_from_web:
            with patch.object(wr, "_save_cache") as mock_save_cache:
                result = wr.retrieve("query")

    mock_check_cache.assert_called()

    # cache is empty, so we call _retrieve_from_web
    mock_retrieve_from_web.assert_called()
    # and save the results to cache
    mock_save_cache.assert_called_with("query", web_docs, cache_index=wr.cache_index, cache_headers=wr.cache_headers)
    assert result == web_docs


@pytest.mark.unit
def test_retrieve_returns_top_k():
    wr = WebRetriever(api_key="", top_k=2)

    with patch.object(wr, "_check_cache", return_value=[]):
        web_docs = [Document("doc1"), Document("doc2"), Document("doc3")]
        with patch.object(wr, "_retrieve_from_web", return_value=web_docs):
            result = wr.retrieve("query")

    assert result == web_docs[:2]


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [1, 3, 6])
def test_top_k_parameter(mock_web_search, top_k):
    web_retriever = WebRetriever(api_key="some_invalid_key", mode="snippets")
    result = web_retriever.retrieve(query="Who is the boyfriend of Olivia Wilde?", top_k=top_k)
    assert len(result) == top_k
    assert all(isinstance(doc, Document) for doc in result)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Please export an env var called SERPERDEV_API_KEY containing the serper.dev API key to run this test.",
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.parametrize("top_k", [2, 4])
def test_top_k_parameter_in_pipeline(top_k):
    # test that WebRetriever top_k param is NOT ignored in a pipeline
    prompt_node = PromptNode(
        "gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_length=256,
        default_prompt_template="question-answering-with-document-scores",
    )

    retriever = WebRetriever(api_key=os.environ.get("SERPERDEV_API_KEY"))

    pipe = Pipeline()

    pipe.add_node(component=retriever, name="WebRetriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="QAwithScoresPrompt", inputs=["WebRetriever"])
    result = pipe.run(query="What year was Obama president", params={"WebRetriever": {"top_k": top_k}})
    assert len(result["results"]) == top_k


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Please export an env var called SERPERDEV_API_KEY containing the serper.dev API key to run this test.",
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.skip
def test_web_retriever_speed():
    retriever = WebRetriever(api_key=os.environ.get("SERPERDEV_API_KEY"), mode="preprocessed_documents")
    result = retriever.retrieve(query="What's the meaning of it all?")
    assert len(result) >= 5
    assert all(isinstance(doc, Document) for doc in result)
