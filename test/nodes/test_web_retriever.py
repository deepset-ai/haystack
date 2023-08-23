import os
from unittest.mock import patch, Mock
from test.conftest import MockDocumentStore
import pytest

from haystack import Document, Pipeline
from haystack.nodes import WebRetriever, PromptNode
from haystack.nodes.retriever.link_content import html_content_handler
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


@pytest.fixture
def mocked_link_content_fetcher_handler_type():
    with patch(
        "haystack.nodes.retriever.link_content.LinkContentFetcher._get_content_type_handler",
        return_value=html_content_handler,
    ):
        yield


@pytest.mark.unit
def test_init_default_parameters():
    retriever = WebRetriever(api_key="test_key")

    assert retriever.top_k == 5
    assert retriever.mode == "snippets"
    assert retriever.preprocessor is None
    assert retriever.cache_document_store is None
    assert retriever.cache_index is None


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["snippets", "raw_documents", "preprocessed_documents"])
@pytest.mark.parametrize("top_k", [1, 5, 7])
def test_retrieve_from_web_all_params(mock_web_search, mode, top_k):
    """
    Test that the retriever returns the correct number of documents in all modes
    """
    search_result_len = len(example_serperdev_response["organic"])
    wr = WebRetriever(api_key="fake_key", top_k=top_k, mode=mode)

    docs = [Document("test" + str(i)) for i in range(search_result_len)]
    with patch("haystack.nodes.retriever.web.WebRetriever._scrape_links", return_value=docs):
        retrieved_docs = wr.retrieve(query="who is the boyfriend of olivia wilde?")

    assert isinstance(retrieved_docs, list)
    assert all(isinstance(doc, Document) for doc in retrieved_docs)
    assert len(retrieved_docs) == top_k


@pytest.mark.unit
def test_retrieve_from_web_invalid_query(mock_web_search):
    """
    Test that the retriever raises an error if the query is invalid
    """
    wr = WebRetriever(api_key="fake_key")
    with pytest.raises(ValueError, match="WebSearch run requires"):
        wr.retrieve("")

    with pytest.raises(ValueError, match="WebSearch run requires"):
        wr.retrieve(None)


@pytest.mark.unit
def test_prepare_links_empty_list():
    """
    Test that the retriever's _prepare_links method returns an empty list if the input is an empty list
    """
    wr = WebRetriever(api_key="fake_key")
    result = wr._prepare_links([])
    assert result == []

    result = wr._prepare_links(None)
    assert result == []


@pytest.mark.unit
def test_scrape_links_empty_list():
    """
    Test that the retriever's _scrape_links method returns an empty list if the input is an empty list
    """
    wr = WebRetriever(api_key="fake_key")
    result = wr._scrape_links([])
    assert result == []


@pytest.mark.unit
def test_scrape_links_with_search_results(
    mocked_requests, mocked_article_extractor, mocked_link_content_fetcher_handler_type
):
    """
    Test that the retriever's _scrape_links method returns a list of Documents if the input is a list of SearchResults
    """
    wr = WebRetriever(api_key="fake_key")

    sr1 = SearchResult("https://pagesix.com", "Some text", 0.43, "1")
    sr2 = SearchResult("https://www.yahoo.com/", "Some text", 0.43, "2")
    fake_search_results = [sr1, sr2]

    result = wr._scrape_links(fake_search_results)

    assert isinstance(result, list)
    assert all(isinstance(r, Document) for r in result)
    assert len(result) == 2


@pytest.mark.unit
def test_scrape_links_with_search_results_with_preprocessor(
    mocked_requests, mocked_article_extractor, mocked_link_content_fetcher_handler_type
):
    """
    Test that the retriever's _scrape_links method returns a list of Documents if the input is a list of SearchResults
    and a preprocessor is provided
    """
    wr = WebRetriever(api_key="fake_key", mode="preprocessed_documents")

    sr1 = SearchResult("https://pagesix.com", "Some text", 0.43, "1")
    sr2 = SearchResult("https://www.yahoo.com/", "Some text", 0.43, "2")
    fake_search_results = [sr1, sr2]

    result = wr._scrape_links(fake_search_results)

    assert isinstance(result, list)
    assert all(isinstance(r, Document) for r in result)
    # the documents from above SearchResult are so small that they will not be split into multiple documents
    # by the preprocessor
    assert len(result) == 2


@pytest.mark.unit
def test_retrieve_checks_cache(mock_web_search):
    """
    Test that the retriever's retrieve method checks the cache
    """
    wr = WebRetriever(api_key="fake_key", mode="preprocessed_documents")

    with patch.object(wr, "_check_cache", return_value=([], [])) as mock_check_cache:
        wr.retrieve("query")

    # assert cache is checked
    mock_check_cache.assert_called()


@pytest.mark.unit
def test_retrieve_no_cache_checks_in_snippet_mode(mock_web_search):
    """
    Test that the retriever's retrieve method does not check the cache if the mode is snippets
    """
    wr = WebRetriever(api_key="fake_key", mode="snippets")

    with patch.object(wr, "_check_cache", return_value=([], [])) as mock_check_cache:
        wr.retrieve("query")

    # assert cache is NOT checked
    mock_check_cache.assert_not_called()


@pytest.mark.unit
def test_retrieve_batch(mock_web_search):
    """
    Test that the retriever's retrieve_batch method returns a list of lists of Documents
    """
    queries = ["query1", "query2"]
    wr = WebRetriever(api_key="fake_key", mode="preprocessed_documents")
    web_docs = [Document("doc1"), Document("doc2"), Document("doc3")]
    with patch("haystack.nodes.retriever.web.WebRetriever._scrape_links", return_value=web_docs):
        result = wr.retrieve_batch(queries)

    assert len(result) == len(queries)

    # check that the result is a list of lists of Documents
    assert all(isinstance(docs, list) for docs in result)
    assert all(isinstance(doc, Document) for docs in result for doc in docs)

    # check that the result is a list of lists of Documents, so that the number of Documents
    # is equal to the number of queries * number of documents retrieved per query
    assert len([doc for docs in result for doc in docs]) == len(web_docs) * len(queries)


@pytest.mark.unit
def test_retrieve_uses_cache(mock_web_search):
    """
    Test that the retriever's retrieve method uses the cache if it is available
    """
    wr = WebRetriever(api_key="fake_key", mode="raw_documents", cache_document_store=MockDocumentStore())

    cached_links = [
        SearchResult("https://pagesix.com", "Some text", 0.43, "1"),
        SearchResult("https://www.yahoo.com/", "Some text", 0.43, "2"),
    ]
    cached_docs = [Document("doc1"), Document("doc2")]
    with patch.object(wr, "_check_cache", return_value=(cached_links, cached_docs)) as mock_check_cache:
        with patch.object(wr, "_save_to_cache") as mock_save_cache:
            with patch.object(wr, "_scrape_links", return_value=[]):
                result = wr.retrieve("query")

    # checking cache is always called
    mock_check_cache.assert_called()

    # cache save is called but with empty list of documents
    mock_save_cache.assert_called()
    assert mock_save_cache.call_args[0][0] == []
    assert result == cached_docs


@pytest.mark.unit
def test_retrieve_saves_to_cache(mock_web_search):
    """
    Test that the retriever's retrieve method saves to the cache if it is available
    """
    wr = WebRetriever(api_key="fake_key", cache_document_store=MockDocumentStore(), mode="preprocessed_documents")
    web_docs = [Document("doc1"), Document("doc2"), Document("doc3")]

    with patch.object(wr, "_save_to_cache") as mock_save_cache:
        with patch.object(wr, "_scrape_links", return_value=web_docs):
            wr.retrieve("query")

    mock_save_cache.assert_called()


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
    """
    Test that the top_k parameter works in the pipeline
    """
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
