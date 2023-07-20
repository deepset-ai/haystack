from typing import List

import pytest

from haystack import Document
from haystack.nodes.ranker.most_diverse import MostDiverseRanker


# Tests that predict method returns a list of Document objects
@pytest.mark.integration
def test_predict_returns_list_of_documents():
    ranker = MostDiverseRanker()
    query = "test query"
    documents = [Document(content="doc1"), Document(content="doc2")]
    result = ranker.predict(query=query, documents=documents)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)


#  Tests that predict method returns the correct number of documents
@pytest.mark.integration
def test_predict_returns_correct_number_of_documents():
    ranker = MostDiverseRanker()
    query = "test query"
    documents = [Document(content="doc1"), Document(content="doc2")]
    result = ranker.predict(query=query, documents=documents, top_k=1)
    assert len(result) == 1


#  Tests that predict method returns documents in the correct order
@pytest.mark.integration
def test_predict_returns_documents_in_correct_order():
    ranker = MostDiverseRanker()
    query = "Sarajevo"
    documents = [
        Document(content="Bosnia"),
        Document(content="Germany"),
        Document(content="cars"),
        Document("Herzegovina"),
    ]
    result = ranker.predict(query=query, documents=documents)
    # the correct most diverse order is: Bosnia, [Germany|cars], Herzegovina,
    assert result[0].content == "Bosnia"
    assert result[1].content == "Germany" or result[1].content == "cars"
    assert result[2].content == "Germany" or result[2].content == "cars"
    assert result[3].content == "Herzegovina"


#  Tests that predict_batch method returns a list of lists of Document objects
@pytest.mark.integration
def test_predict_batch_returns_list_of_lists_of_documents():
    ranker = MostDiverseRanker()
    queries = ["test query 1", "test query 2"]
    documents = [
        [Document(content="doc1"), Document(content="doc2")],
        [Document(content="doc3"), Document(content="doc4")],
    ]
    result: List[List[Document]] = ranker.predict_batch(queries=queries, documents=documents)
    assert isinstance(result, list)
    assert all(isinstance(docs, list) for docs in result)
    assert all(isinstance(doc, Document) for docs in result for doc in docs)


#  Tests that predict_batch method returns the correct number of documents
@pytest.mark.integration
def test_predict_batch_returns_correct_number_of_documents():
    ranker = MostDiverseRanker()
    queries = ["test query 1", "test query 2"]
    documents = [
        [Document(content="doc1"), Document(content="doc2")],
        [Document(content="doc3"), Document(content="doc4")],
    ]
    result: List[List[Document]] = ranker.predict_batch(queries=queries, documents=documents, top_k=1)
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 1


#  Tests that predict_batch method returns documents in the correct order
@pytest.mark.integration
def test_predict_batch_returns_documents_in_correct_order():
    ranker = MostDiverseRanker()
    queries = ["Berlin", "Paris"]
    documents = [
        [Document(content="Germany"), Document(content="Munich"), Document(content="agriculture")],
        [Document(content="France"), Document(content="Space exploration"), Document(content="Eiffel Tower")],
    ]
    result: List[List[Document]] = ranker.predict_batch(queries=queries, documents=documents)
    most_diverse0 = result[0]
    most_diverse1 = result[1]

    # check the correct most diverse order are in batches
    assert most_diverse0[0].content == "Germany"
    assert most_diverse0[1].content == "agriculture"
    assert most_diverse0[2].content == "Munich"

    assert most_diverse1[0].content == "France"
    assert most_diverse1[1].content == "Space exploration"
    assert most_diverse1[2].content == "Eiffel Tower"


#  Tests that predict method raises ValueError if query is empty
@pytest.mark.integration
def test_predict_raises_value_error_if_query_is_empty():
    ranker = MostDiverseRanker()
    query = ""
    documents = [Document(content="doc1"), Document(content="doc2")]
    with pytest.raises(ValueError):
        ranker.predict(query=query, documents=documents)


#  Tests that predict method raises ValueError if documents is empty
@pytest.mark.integration
def test_predict_raises_value_error_if_documents_is_empty():
    ranker = MostDiverseRanker()
    query = "test query"
    documents = []
    with pytest.raises(ValueError):
        ranker.predict(query=query, documents=documents)


#  Tests that predict_batch method raises ValueError if queries is empty
@pytest.mark.integration
def test_predict_batch_raises_value_error_if_queries_is_empty():
    ranker = MostDiverseRanker()
    queries = []
    documents = [
        [Document(content="doc1"), Document(content="doc2")],
        [Document(content="doc3"), Document(content="doc4")],
    ]
    with pytest.raises(ValueError):
        ranker.predict_batch(queries=queries, documents=documents)


#  Tests that predict_batch method raises ValueError if documents is empty
@pytest.mark.integration
def test_predict_batch_raises_value_error_if_documents_is_empty():
    ranker = MostDiverseRanker()
    queries = ["test query 1", "test query 2"]
    documents = []
    with pytest.raises(ValueError):
        ranker.predict_batch(queries=queries, documents=documents)
