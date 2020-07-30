import pytest
from haystack.database.base import BaseDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever


def test_add_eval_data(document_store):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename="samples/squad/small.json", doc_index="test_eval_document", label_index="test_feedback")

    assert document_store.get_document_count(index="test_eval_document") == 87
    assert document_store.get_label_count(index="test_feedback") == 881

    # test documents
    docs = document_store.get_all_documents(index="test_eval_document")
    assert docs[0].text[:10] == "The Norman"
    assert docs[0].meta["name"] == "Normans"
    assert len(docs[0].meta.keys()) == 1

    # test labels
    labels = document_store.get_all_labels(index="test_feedback")
    assert labels[0].answer == "France"
    assert labels[0].no_answer == False
    assert labels[0].is_correct_answer == True
    assert labels[0].is_correct_document == True
    assert labels[0].question == 'In what country is Normandy located?'
    assert labels[0].origin == "gold_label"
    assert labels[0].offset_start_in_doc == 159

    # check combination
    assert labels[0].document_id == docs[0].id
    start = labels[0].offset_start_in_doc
    end = start+len(labels[0].answer)
    assert docs[0].text[start:end] == "France"

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")


@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_eval_reader(reader, document_store: BaseDocumentStore):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename="samples/squad/tiny.json", doc_index="test_eval_document", label_index="test_feedback")
    assert document_store.get_document_count(index="test_eval_document") == 2
    # eval reader
    reader_eval_results = reader.eval(document_store=document_store, label_index="test_feedback",
                                      doc_index="test_eval_document", device="cpu")
    assert reader_eval_results["f1"] > 0.65
    assert reader_eval_results["f1"] < 0.67
    assert reader_eval_results["EM"] == 0.5
    assert reader_eval_results["top_n_accuracy"] == 1.0

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")


@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("open_domain", [True, False])
def test_eval_elastic_retriever(document_store: BaseDocumentStore, open_domain):
    retriever = ElasticsearchRetriever(document_store=document_store)

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename="samples/squad/tiny.json", doc_index="test_eval_document", label_index="test_feedback")
    assert document_store.get_document_count(index="test_eval_document") == 2

    # eval retriever
    results = retriever.eval(top_k=1, label_index="test_feedback", doc_index="test_eval_document", open_domain=open_domain)
    assert results["recall"] == 1.0
    assert results["map"] == 1.0

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
