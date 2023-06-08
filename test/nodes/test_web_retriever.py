import os
from typing import Dict, Tuple


import pytest
import requests
from boilerpy3.extractors import ArticleExtractor

from haystack import Document, Pipeline
from haystack.nodes import WebSearch, WebRetriever, PromptNode


@pytest.mark.unit
def test_web_retriever_mode_raw_documents(monkeypatch):
    expected_search_results = {
        "documents": [
            Document(
                content="Eddard Stark",
                score=0.9090909090909091,
                meta={"title": "Eddard Stark", "link": "", "score": 0.9090909090909091},
                id_hash_keys=["content"],
                id="f408db6de8de0ffad0cb47cf8830dbb8",
            ),
            Document(
                content="The most likely answer for the clue is NED. How many solutions does Arya Stark's Father have? With crossword-solver.io you will find 1 solutions. We use ...",
                score=0.09090909090909091,
                meta={
                    "title": "Arya Stark's Father - Crossword Clue Answers",
                    "link": "https://crossword-solver.io/clue/arya-stark%27s-father/",
                    "position": 1,
                    "score": 0.09090909090909091,
                },
                id_hash_keys=["content"],
                id="51779277acf94cf90e7663db137c0732",
            ),
        ]
    }

    def mock_web_search_run(self, query: str) -> Tuple[Dict, str]:
        return expected_search_results, "output_1"

    class MockResponse:
        def __init__(self, text, status_code):
            self.text = text
            self.status_code = status_code

    def get(url, headers, timeout):
        return MockResponse("mocked", 200)

    def get_content(self, text: str) -> str:
        return "What are the top solutions for\nArya Stark's Father\nWe found 1 solutions for\nArya Stark's Father\n.The top solutions is determined by popularity, ratings and frequency of searches. The most likely answer for the clue is NED..."

    monkeypatch.setattr(WebSearch, "run", mock_web_search_run)
    monkeypatch.setattr(ArticleExtractor, "get_content", get_content)
    monkeypatch.setattr(requests, "get", get)

    web_retriever = WebRetriever(api_key="", top_search_results=2, mode="raw_documents")
    result = web_retriever.retrieve(query="Who is the father of Arya Stark?")
    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert (
        result[0].content
        == "What are the top solutions for\nArya Stark's Father\nWe found 1 solutions for\nArya Stark's Father\n.The top solutions is determined by popularity, ratings and frequency of searches. The most likely answer for the clue is NED..."
    )
    assert result[0].score == None
    assert result[0].meta["url"] == "https://crossword-solver.io/clue/arya-stark%27s-father/"
    # Only preprocessed docs but not raw docs should have the _split_id field
    assert "_split_id" not in result[0].meta


@pytest.mark.unit
def test_web_retriever_mode_preprocessed_documents(monkeypatch):
    expected_search_results = {
        "documents": [
            Document(
                content="Eddard Stark",
                score=0.9090909090909091,
                meta={"title": "Eddard Stark", "link": "", "score": 0.9090909090909091},
                id_hash_keys=["content"],
                id="f408db6de8de0ffad0cb47cf8830dbb8",
            ),
            Document(
                content="The most likely answer for the clue is NED. How many solutions does Arya Stark's Father have? With crossword-solver.io you will find 1 solutions. We use ...",
                score=0.09090909090909091,
                meta={
                    "title": "Arya Stark's Father - Crossword Clue Answers",
                    "link": "https://crossword-solver.io/clue/arya-stark%27s-father/",
                    "position": 1,
                    "score": 0.09090909090909091,
                },
                id_hash_keys=["content"],
                id="51779277acf94cf90e7663db137c0732",
            ),
        ]
    }

    def mock_web_search_run(self, query: str) -> Tuple[Dict, str]:
        return expected_search_results, "output_1"

    class MockResponse:
        def __init__(self, text, status_code):
            self.text = text
            self.status_code = status_code

    def get(url, headers, timeout):
        return MockResponse("mocked", 200)

    def get_content(self, text: str) -> str:
        return "What are the top solutions for\nArya Stark's Father\nWe found 1 solutions for\nArya Stark's Father\n.The top solutions is determined by popularity, ratings and frequency of searches. The most likely answer for the clue is NED..."

    monkeypatch.setattr(WebSearch, "run", mock_web_search_run)
    monkeypatch.setattr(ArticleExtractor, "get_content", get_content)
    monkeypatch.setattr(requests, "get", get)

    web_retriever = WebRetriever(api_key="", top_search_results=2, mode="preprocessed_documents")
    result = web_retriever.retrieve(query="Who is the father of Arya Stark?")
    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert (
        result[0].content
        == "What are the top solutions for\nArya Stark's Father\nWe found 1 solutions for\nArya Stark's Father\n.The top solutions is determined by popularity, ratings and frequency of searches. The most likely answer for the clue is NED..."
    )
    assert result[0].score == None
    assert result[0].meta["url"] == "https://crossword-solver.io/clue/arya-stark%27s-father/"
    assert result[0].meta["_split_id"] == 0


@pytest.mark.unit
def test_web_retriever_mode_snippets(monkeypatch):
    expected_search_results = {
        "documents": [
            Document(
                content="Eddard Stark",
                score=0.9090909090909091,
                meta={"title": "Eddard Stark", "link": "", "score": 0.9090909090909091},
                id_hash_keys=["content"],
                id="f408db6de8de0ffad0cb47cf8830dbb8",
            ),
            Document(
                content="The most likely answer for the clue is NED. How many solutions does Arya Stark's Father have? With crossword-solver.io you will find 1 solutions. We use ...",
                score=0.09090909090909091,
                meta={
                    "title": "Arya Stark's Father - Crossword Clue Answers",
                    "link": "https://crossword-solver.io/clue/arya-stark%27s-father/",
                    "position": 1,
                    "score": 0.09090909090909091,
                },
                id_hash_keys=["content"],
                id="51779277acf94cf90e7663db137c0732",
            ),
        ]
    }

    def mock_web_search_run(self, query: str) -> Tuple[Dict, str]:
        return expected_search_results, "output_1"

    monkeypatch.setattr(WebSearch, "run", mock_web_search_run)
    web_retriever = WebRetriever(api_key="", top_search_results=2)
    result = web_retriever.retrieve(query="Who is the father of Arya Stark?")
    assert result == expected_search_results["documents"]


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
