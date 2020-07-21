import pytest
from typing import Type
from haystack.database.base import BaseDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever

@pytest.mark.parametrize("reader", [("farm")], indirect=True)
@pytest.mark.parametrize("document_store", [("elasticsearch")], indirect=True)
def test_eval_reader(reader, document_store: Type[BaseDocumentStore]):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="eval_document")
    document_store.delete_all_documents(index="feedback")
    document_store.add_eval_data(filename="samples/squad/tiny.json", doc_index="eval_document", label_index="feedback")
    assert document_store.get_document_count(index="eval_document") == 2
    # eval reader
    reader_eval_results = reader.eval(document_store=document_store, device="cpu")
    assert reader_eval_results["f1"] > 0.65
    assert reader_eval_results["f1"] < 0.67
    assert reader_eval_results["EM"] == 0.5
    assert reader_eval_results["top_n_accuracy"] == 1.0

@pytest.mark.parametrize("document_store", [("elasticsearch")], indirect=True)
def test_eval_elastic_retriever(document_store: Type[BaseDocumentStore]):
    retriever = ElasticsearchRetriever(document_store=document_store)

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="eval_document")
    document_store.delete_all_documents(index="feedback")
    document_store.add_eval_data(filename="samples/squad/tiny.json", doc_index="eval_document", label_index="feedback")
    assert document_store.get_document_count(index="eval_document") == 2
    # eval retriever
    results = retriever.eval(top_k=1)
    assert results["recall"] == 1.0
    assert results["map"] == 1.0
