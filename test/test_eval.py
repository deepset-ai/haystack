import pytest
from haystack.document_store.base import BaseDocumentStore
from haystack.finder import Finder


@pytest.mark.elasticsearch
def test_add_eval_data(document_store):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename="samples/squad/small.json", doc_index="test_eval_document", label_index="test_feedback")

    assert document_store.get_document_count(index="test_eval_document") == 87
    assert document_store.get_label_count(index="test_feedback") == 1214

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


@pytest.mark.elasticsearch
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


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("open_domain", [True, False])
@pytest.mark.parametrize("retriever", ["elasticsearch"], indirect=True)
def test_eval_elastic_retriever(document_store: BaseDocumentStore, open_domain, retriever):
    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename="samples/squad/tiny.json", doc_index="test_eval_document", label_index="test_feedback")
    assert document_store.get_document_count(index="test_eval_document") == 2

    # eval retriever
    results = retriever.eval(top_k=1, label_index="test_feedback", doc_index="test_eval_document", open_domain=open_domain)
    assert results["recall"] == 1.0
    assert results["mrr"] == 1.0
    if not open_domain:
        assert results["map"] == 1.0

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever", ["elasticsearch"], indirect=True)
def test_eval_finder(document_store: BaseDocumentStore, reader, retriever):
    finder = Finder(reader=reader, retriever=retriever)

    # add eval data (SQUAD format)
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")
    document_store.add_eval_data(filename="samples/squad/tiny.json", doc_index="test_eval_document", label_index="test_feedback")
    assert document_store.get_document_count(index="test_eval_document") == 2

    # eval finder
    results = finder.eval(label_index="test_feedback", doc_index="test_eval_document", top_k_retriever=1, top_k_reader=5)
    assert results["retriever_recall"] == 1.0
    assert results["retriever_map"] == 1.0
    assert abs(results["reader_topk_f1"] - 0.66666) < 0.001
    assert abs(results["reader_topk_em"] - 0.5) < 0.001
    assert abs(results["reader_topk_accuracy"] - 1) < 0.001
    assert results["reader_top1_f1"] <= results["reader_topk_f1"]
    assert results["reader_top1_em"] <= results["reader_topk_em"]
    assert results["reader_top1_accuracy"] <= results["reader_topk_accuracy"]

    # batch eval finder
    results_batch = finder.eval_batch(label_index="test_feedback", doc_index="test_eval_document", top_k_retriever=1,
                          top_k_reader=5)
    assert results_batch["retriever_recall"] == 1.0
    assert results_batch["retriever_map"] == 1.0
    assert results_batch["reader_top1_f1"] == results["reader_top1_f1"]
    assert results_batch["reader_top1_em"] == results["reader_top1_em"]
    assert results_batch["reader_topk_accuracy"] == results["reader_topk_accuracy"]

    # clean up
    document_store.delete_all_documents(index="test_eval_document")
    document_store.delete_all_documents(index="test_feedback")