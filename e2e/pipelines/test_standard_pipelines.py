import os

import pytest

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever.web import WebRetriever
from haystack.pipelines import (
    Pipeline,
    FAQPipeline,
    DocumentSearchPipeline,
    MostSimilarDocumentsPipeline,
    WebQAPipeline,
    SearchSummarizationPipeline,
)
from haystack.nodes import EmbeddingRetriever, PromptNode, BM25Retriever, TransformersSummarizer
from haystack.schema import Document


def test_faq_pipeline():
    documents = [
        {"content": f"How to test module-{i}?", "meta": {"source": f"wiki{i}", "answer": f"Using tests for module-{i}"}}
        for i in range(1, 6)
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = FAQPipeline(retriever=retriever)

    output = pipeline.run(query="How to test this?", params={"Retriever": {"top_k": 3}})
    assert len(output["answers"]) == 3
    assert output["query"].startswith("How to")
    assert output["answers"][0].answer.startswith("Using tests")

    output = pipeline.run(
        query="How to test this?", params={"Retriever": {"filters": {"source": ["wiki2"]}, "top_k": 5}}
    )
    assert len(output["answers"]) == 1


def test_document_search_pipeline():
    documents = [
        {"content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever=retriever)
    output = pipeline.run(query="How to test this?", params={"top_k": 4})
    assert len(output.get("documents", [])) == 4

    output = pipeline.run(query="How to test this?", params={"filters": {"source": ["wiki2"]}, "top_k": 5})
    assert len(output["documents"]) == 1


def test_most_similar_documents_pipeline():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run(document_ids=docs_id)

    assert len(list_of_documents[0]) > 1
    assert isinstance(list_of_documents, list)
    assert len(list_of_documents) == len(docs_id)

    for another_list in list_of_documents:
        assert isinstance(another_list, list)
        for document in another_list:
            assert isinstance(document, Document)
            assert isinstance(document.id, str)
            assert isinstance(document.content, str)


def test_most_similar_documents_pipeline_with_filters():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore()
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    filters = {"source": ["wiki3", "wiki4", "wiki5"]}
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run(document_ids=docs_id, filters=filters)

    assert len(list_of_documents[0]) > 1
    assert isinstance(list_of_documents, list)
    assert len(list_of_documents) == len(docs_id)

    for another_list in list_of_documents:
        assert isinstance(another_list, list)
        for document in another_list:
            assert isinstance(document, Document)
            assert isinstance(document.id, str)
            assert isinstance(document.content, str)
            assert document.meta["source"] in ["wiki3", "wiki4", "wiki5"]


def test_query_and_indexing_pipeline(samples_path):
    # test correct load of indexing pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        samples_path / "pipelines" / "test.haystack-pipeline.yml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(file_paths=samples_path / "pipelines" / "sample_pdf_1.pdf")
    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        samples_path / "pipelines" / "test.haystack-pipeline.yml", pipeline_name="query_pipeline"
    )
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"Retriever": {"top_k": 2}, "Reader": {"top_k": 1}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert "_debug" not in prediction.keys()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.skipif(
    not os.environ.get("SERPERDEV_API_KEY", None),
    reason="Please export an env var called SERPERDEV_API_KEY containing the SerperDev key to run this test.",
)
def test_webqa_pipeline():
    search_key = os.environ.get("SERPERDEV_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    pn = PromptNode(
        "text-davinci-003",
        api_key=openai_key,
        max_length=256,
        default_prompt_template="question-answering-with-document-scores",
    )
    web_retriever = WebRetriever(api_key=search_key, top_search_results=2)
    pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)
    result = pipeline.run(query="Who is the father of Arya Stark?")
    assert isinstance(result, dict)
    assert len(result["results"]) == 1
    answer = result["results"][0]
    assert "Stark" in answer or "NED" in answer


def test_faq_pipeline_batch():
    documents = [
        {"content": "How to test module-1?", "meta": {"source": "wiki1", "answer": "Using tests for module-1"}},
        {"content": "How to test module-2?", "meta": {"source": "wiki2", "answer": "Using tests for module-2"}},
        {"content": "How to test module-3?", "meta": {"source": "wiki3", "answer": "Using tests for module-3"}},
        {"content": "How to test module-4?", "meta": {"source": "wiki4", "answer": "Using tests for module-4"}},
        {"content": "How to test module-5?", "meta": {"source": "wiki5", "answer": "Using tests for module-5"}},
    ]
    document_store = InMemoryDocumentStore(embedding_dim=384)
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = FAQPipeline(retriever=retriever)

    output = pipeline.run_batch(queries=["How to test this?", "How to test this?"], params={"Retriever": {"top_k": 3}})
    assert len(output["answers"]) == 2  # 2 queries
    assert len(output["answers"][0]) == 3  # 3 answers per query
    assert output["queries"][0].startswith("How to")
    assert output["answers"][0][0].answer.startswith("Using tests")


def test_document_search_pipeline_batch():
    documents = [
        {"content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore(embedding_dim=384)
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever=retriever)
    output = pipeline.run_batch(queries=["How to test this?", "How to test this?"], params={"top_k": 4})
    assert len(output["documents"]) == 2  # 2 queries
    assert len(output["documents"][0]) == 4  # 4 docs per query


def test_most_similar_documents_pipeline_batch():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore(embedding_dim=384)
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run_batch(document_ids=docs_id)

    assert len(list_of_documents[0]) > 1
    assert isinstance(list_of_documents, list)
    assert len(list_of_documents) == len(docs_id)

    for another_list in list_of_documents:
        assert isinstance(another_list, list)
        for document in another_list:
            assert isinstance(document, Document)
            assert isinstance(document.id, str)
            assert isinstance(document.content, str)


def test_most_similar_documents_pipeline_with_filters_batch():
    documents = [
        {"id": "a", "content": "Sample text for document-1", "meta": {"source": "wiki1"}},
        {"id": "b", "content": "Sample text for document-2", "meta": {"source": "wiki2"}},
        {"content": "Sample text for document-3", "meta": {"source": "wiki3"}},
        {"content": "Sample text for document-4", "meta": {"source": "wiki4"}},
        {"content": "Sample text for document-5", "meta": {"source": "wiki5"}},
    ]
    document_store = InMemoryDocumentStore(embedding_dim=384)
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_store = InMemoryDocumentStore(embedding_dim=384)
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

    docs_id: list = ["a", "b"]
    filters = {"source": ["wiki3", "wiki4", "wiki5"]}
    pipeline = MostSimilarDocumentsPipeline(document_store=document_store)
    list_of_documents = pipeline.run_batch(document_ids=docs_id, filters=filters)

    assert len(list_of_documents[0]) > 1
    assert isinstance(list_of_documents, list)
    assert len(list_of_documents) == len(docs_id)

    for another_list in list_of_documents:
        assert isinstance(another_list, list)
        for document in another_list:
            assert isinstance(document, Document)
            assert isinstance(document.id, str)
            assert isinstance(document.content, str)
            assert document.meta["source"] in ["wiki3", "wiki4", "wiki5"]


def test_summarization_pipeline():
    docs = [
        Document(
            content="""
    PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions.
    The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected
    by the shutoffs which were expected to last through at least midday tomorrow.
    """
        ),
        Document(
            content="""
    The Eiffel Tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest
    structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction,
    the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a
    title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first
    structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower
    in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel
    Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    """
        ),
    ]
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-6", use_gpu=False)

    ds = InMemoryDocumentStore(use_bm25=True)
    retriever = BM25Retriever(document_store=ds)
    ds.write_documents(docs)

    query = "Eiffel Tower"
    pipeline = SearchSummarizationPipeline(retriever=retriever, summarizer=summarizer, return_in_answer_format=True)
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 1}})
    answers = output["answers"]
    assert len(answers) == 1
    assert "The Eiffel Tower is one of the world's tallest structures." == answers[0]["answer"].strip()


def test_summarization_pipeline_one_summary():
    split_docs = [
        Document(
            content="""
    The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.
    Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the
    Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler
    Building in New York City was finished in 1930.
    """
        ),
        Document(
            content="""
    It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the
    top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters,
    the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    """
        ),
    ]
    ds = InMemoryDocumentStore(use_bm25=True)
    retriever = BM25Retriever(document_store=ds)
    ds.write_documents(split_docs)
    summarizer = TransformersSummarizer(model_name_or_path="sshleifer/distilbart-xsum-12-6", use_gpu=False)

    query = "Eiffel Tower"
    pipeline = SearchSummarizationPipeline(
        retriever=retriever, summarizer=summarizer, generate_single_summary=True, return_in_answer_format=True
    )
    output = pipeline.run(query=query, params={"Retriever": {"top_k": 2}})
    answers = output["answers"]
    assert len(answers) == 1
    assert answers[0]["answer"].strip() == "The Eiffel Tower was built in 1924 in Paris, France."
