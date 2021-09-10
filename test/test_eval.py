import pytest
from haystack.document_store.base import BaseDocumentStore
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.eval import EvalAnswers, EvalDocuments
from haystack import Pipeline

@pytest.mark.parametrize("batch_size", [None, 20])
@pytest.mark.elasticsearch
def test_add_eval_data(document_store, batch_size):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename="samples/squad/small.json",
        doc_index="haystack_test_eval_document",
        label_index="haystack_test_feedback",
        batch_size=batch_size,
    )

    assert document_store.get_document_count(index="haystack_test_eval_document") == 87
    assert document_store.get_label_count(index="haystack_test_feedback") == 1214

    # test documents
    docs = document_store.get_all_documents(index="haystack_test_eval_document", filters={"name": ["Normans"]})
    assert docs[0].meta["name"] == "Normans"
    assert len(docs[0].meta.keys()) == 1

    # test labels
    labels = document_store.get_all_labels(index="haystack_test_feedback")
    label = None
    for l in labels:
        if l.question == "In what country is Normandy located?":
            label = l
            break
    assert label.answer == "France"
    assert label.no_answer == False
    assert label.is_correct_answer == True
    assert label.is_correct_document == True
    assert label.question == "In what country is Normandy located?"
    assert label.origin == "gold_label"
    assert label.offset_start_in_doc == 159

    # check combination
    doc = document_store.get_document_by_id(label.document_id, index="haystack_test_eval_document")
    start = label.offset_start_in_doc
    end = start + len(label.answer)
    assert doc.text[start:end] == "France"


@pytest.mark.elasticsearch
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_eval_reader(reader, document_store: BaseDocumentStore):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename="samples/squad/tiny.json",
        doc_index="haystack_test_eval_document",
        label_index="haystack_test_feedback",
    )
    assert document_store.get_document_count(index="haystack_test_eval_document") == 2
    # eval reader
    reader_eval_results = reader.eval(
        document_store=document_store,
        label_index="haystack_test_feedback",
        doc_index="haystack_test_eval_document",
        device="cpu",
    )
    assert reader_eval_results["f1"] > 66.65
    assert reader_eval_results["f1"] < 66.67
    assert reader_eval_results["EM"] == 50
    assert reader_eval_results["top_n_accuracy"] == 100.0


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("open_domain", [True, False])
@pytest.mark.parametrize("retriever", ["elasticsearch"], indirect=True)
def test_eval_elastic_retriever(document_store: BaseDocumentStore, open_domain, retriever):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename="samples/squad/tiny.json",
        doc_index="haystack_test_eval_document",
        label_index="haystack_test_feedback",
    )
    assert document_store.get_document_count(index="haystack_test_eval_document") == 2

    # eval retriever
    results = retriever.eval(
        top_k=1, label_index="haystack_test_feedback", doc_index="haystack_test_eval_document", open_domain=open_domain
    )
    assert results["recall"] == 1.0
    assert results["mrr"] == 1.0
    if not open_domain:
        assert results["map"] == 1.0


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
@pytest.mark.parametrize("retriever", ["elasticsearch"], indirect=True)
def test_eval_pipeline(document_store: BaseDocumentStore, reader, retriever):
    # add eval data (SQUAD format)
    document_store.add_eval_data(
        filename="samples/squad/tiny.json",
        doc_index="haystack_test_eval_document",
        label_index="haystack_test_feedback",
    )

    labels = document_store.get_all_labels_aggregated(index="haystack_test_feedback")

    eval_retriever = EvalDocuments()
    eval_reader = EvalAnswers(sas_model="sentence-transformers/paraphrase-MiniLM-L3-v2",debug=True)
    eval_reader_cross = EvalAnswers(sas_model="cross-encoder/stsb-TinyBERT-L-4",debug=True)
    eval_reader_vanila = EvalAnswers()

    assert document_store.get_document_count(index="haystack_test_eval_document") == 2
    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=eval_retriever, name="EvalDocuments", inputs=["ESRetriever"])
    p.add_node(component=reader, name="QAReader", inputs=["EvalDocuments"])
    p.add_node(component=eval_reader, name="EvalAnswers", inputs=["QAReader"])
    p.add_node(component=eval_reader_cross, name="EvalAnswers_cross", inputs=["QAReader"])
    p.add_node(component=eval_reader_vanila, name="EvalAnswers_vanilla", inputs=["QAReader"])
    for l in labels:
        res = p.run(
            query=l.question,
            labels=l,
            params={"index": "haystack_test_eval_document"}
        )
    assert eval_retriever.recall == 1.0
    assert round(eval_reader.top_k_f1, 4) == 0.8333
    assert eval_reader.top_k_em == 0.5
    assert round(eval_reader.top_k_sas, 3) == 0.800
    assert round(eval_reader_cross.top_k_sas, 3) == 0.671
    assert eval_reader.top_k_em == eval_reader_vanila.top_k_em

@pytest.mark.elasticsearch
def test_eval_data_split_word(document_store):
    # splitting by word
    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by="word",
        split_length=4,
        split_overlap=0,
        split_respect_sentence_boundary=False,
    )

    document_store.add_eval_data(
        filename="samples/squad/tiny.json",
        doc_index="haystack_test_eval_document",
        label_index="haystack_test_feedback",
        preprocessor=preprocessor,
    )
    labels = document_store.get_all_labels_aggregated(index="haystack_test_feedback")
    docs = document_store.get_all_documents(index="haystack_test_eval_document")
    assert len(docs) == 5
    assert len(set(labels[0].multiple_document_ids)) == 2


@pytest.mark.elasticsearch
def test_eval_data_split_passage(document_store):
    # splitting by passage
    preprocessor = PreProcessor(
        clean_empty_lines=False,
        clean_whitespace=False,
        clean_header_footer=False,
        split_by="passage",
        split_length=1,
        split_overlap=0,
        split_respect_sentence_boundary=False
    )

    document_store.add_eval_data(
        filename="samples/squad/tiny_passages.json",
        doc_index="haystack_test_eval_document",
        label_index="haystack_test_feedback",
        preprocessor=preprocessor,
    )
    docs = document_store.get_all_documents(index="haystack_test_eval_document")
    assert len(docs) == 2
    assert len(docs[1].text) == 56