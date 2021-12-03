import pytest
from haystack.document_stores.base import BaseDocumentStore
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.evaluator import EvalAnswers, EvalDocuments
from haystack.nodes.query_classifier.transformers import TransformersQueryClassifier
from haystack.nodes.retriever.dense import DensePassageRetriever
from haystack.nodes.retriever.sparse import ElasticsearchRetriever
from haystack.pipelines.base import Pipeline
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines.standard_pipelines import DocumentSearchPipeline, FAQPipeline, RetrieverQuestionGenerationPipeline, TranslationWrapperPipeline
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel, Span


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
@pytest.mark.parametrize("batch_size", [None, 20])
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
        if l.query == "In what country is Normandy located?":
            label = l
            break
    assert label.answer.answer == "France"
    assert label.no_answer == False
    assert label.is_correct_answer == True
    assert label.is_correct_document == True
    assert label.query == "In what country is Normandy located?"
    assert label.origin == "gold-label"
    assert label.answer.offsets_in_document[0].start == 159
    assert label.answer.context[label.answer.offsets_in_context[0].start:label.answer.offsets_in_context[0].end] == "France"
    assert label.answer.document_id == label.document.id

    # check combination
    doc = document_store.get_document_by_id(label.document.id, index="haystack_test_eval_document")
    start = label.answer.offsets_in_document[0].start
    end = label.answer.offsets_in_document[0].end
    assert end == start + len(label.answer.answer)
    assert doc.content[start:end] == "France"


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
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


# TODO simplify with a mock retriever and make it independent of elasticsearch documentstore
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

    labels = document_store.get_all_labels_aggregated(index="haystack_test_feedback",
                                                      drop_negative_labels=True,
                                                      drop_no_answers=False)

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
            query=l.query,
            labels=l,
            params={"ESRetriever":{"index": "haystack_test_eval_document"}}
        )
    assert eval_retriever.recall == 1.0
    assert round(eval_reader.top_k_f1, 4) == 0.8333
    assert eval_reader.top_k_em == 0.5
    assert round(eval_reader.top_k_sas, 3) == 0.800
    assert round(eval_reader_cross.top_k_sas, 3) == 0.671
    assert eval_reader.top_k_em == eval_reader_vanila.top_k_em


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
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
    assert len(set(labels[0].document_ids)) == 2


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus"], indirect=True)
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
    assert len(docs[1].content) == 56


EVAL_LABELS = [
        MultiLabel(labels=[Label(query="Who lives in Berlin?", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")]),
        MultiLabel(labels=[Label(query="Who lives in Munich?", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='something_else', content_type="text", content='My name is Carla and I live in Munich'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
    ]

@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval(reader, retriever_with_docs, tmp_path):
    labels = EVAL_LABELS[:1]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result = pipeline.eval(
        labels=labels,
        params={"Retriever": {"top_k": 5}}, 
    )

    metrics = eval_result.calculate_metrics()

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    assert reader_result[reader_result['rank'] == 1]["answer"].iloc[0] in reader_result[reader_result['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_result[retriever_result['rank'] == 1]["document_id"].iloc[0] in retriever_result[retriever_result['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0/3
    assert metrics["Retriever"]["map"] == 1.0

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)
    metrics = saved_eval_result.calculate_metrics()

    assert reader_result[reader_result['rank'] == 1]["answer"].iloc[0] in reader_result[reader_result['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_result[retriever_result['rank'] == 1]["document_id"].iloc[0] in retriever_result[retriever_result['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["recall_multi_hit"] == 1.0
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0/3
    assert metrics["Retriever"]["map"] == 1.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_multiple_queries(reader, retriever_with_docs, tmp_path):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    reader_result = eval_result["Reader"]
    retriever_result = eval_result["Retriever"]

    reader_berlin = reader_result[reader_result['query'] == "Who lives in Berlin?"]
    reader_munich = reader_result[reader_result['query'] == "Who lives in Munich?"]

    retriever_berlin = retriever_result[retriever_result['query'] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result['query'] == "Who lives in Munich?"]

    assert reader_berlin[reader_berlin['rank'] == 1]["answer"].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]["document_id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]["answer"].iloc[0] not in reader_munich[reader_munich['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["document_id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6

    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)
    metrics = saved_eval_result.calculate_metrics()

    assert reader_berlin[reader_berlin['rank'] == 1]["answer"].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]["document_id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]["answer"].iloc[0] not in reader_munich[reader_munich['rank'] == 1]["gold_answers"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["document_id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_sas(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}, 
        sas_model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    metrics = eval_result.calculate_metrics()

    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6
    assert "sas" in metrics["Reader"]
    assert metrics["Reader"]["sas"] == pytest.approx(1.0)


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_doc_relevance_col(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}, 
    )

    metrics = eval_result.calculate_metrics(doc_relevance_col="gold_id_or_answer_match")

    assert metrics["Retriever"]["mrr"] == 1.0
    assert metrics["Retriever"]["map"] == 0.75
    assert metrics["Retriever"]["recall_multi_hit"] == 0.75
    assert metrics["Retriever"]["recall_single_hit"] == 1.0
    assert metrics["Retriever"]["precision"] == 1.0/3


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_simulated_top_k_reader(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}},
        sas_model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1)
    
    assert metrics_top_1["Reader"]["exact_match"] == 0.5
    assert metrics_top_1["Reader"]["f1"] == 0.5
    assert metrics_top_1["Reader"]["sas"] == pytest.approx(0.6208, abs=1e-4)
    assert metrics_top_1["Retriever"]["mrr"] == 0.5
    assert metrics_top_1["Retriever"]["map"] == 0.5
    assert metrics_top_1["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_1["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_1["Retriever"]["precision"] == 1.0/6

    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_reader=2)
    
    assert metrics_top_2["Reader"]["exact_match"] == 0.5
    assert metrics_top_2["Reader"]["f1"] == 0.5
    assert metrics_top_2["Reader"]["sas"] == pytest.approx(0.7192, abs=1e-4)
    assert metrics_top_2["Retriever"]["mrr"] == 0.5
    assert metrics_top_2["Retriever"]["map"] == 0.5
    assert metrics_top_2["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_2["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_2["Retriever"]["precision"] == 1.0/6

    metrics_top_3 = eval_result.calculate_metrics(simulated_top_k_reader=3)
    
    assert metrics_top_3["Reader"]["exact_match"] == 1.0
    assert metrics_top_3["Reader"]["f1"] == 1.0
    assert metrics_top_3["Reader"]["sas"] == pytest.approx(1.0)
    assert metrics_top_3["Retriever"]["mrr"] == 0.5
    assert metrics_top_3["Retriever"]["map"] == 0.5
    assert metrics_top_3["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_3["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_3["Retriever"]["precision"] == 1.0/6
    

@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_simulated_top_k_retriever(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics_top_10 = eval_result.calculate_metrics()

    assert metrics_top_10["Reader"]["exact_match"] == 1.0
    assert metrics_top_10["Reader"]["f1"] == 1.0
    assert metrics_top_10["Retriever"]["mrr"] == 0.5
    assert metrics_top_10["Retriever"]["map"] == 0.5
    assert metrics_top_10["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_10["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_10["Retriever"]["precision"] == 1.0/6

    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_retriever=1)
    
    assert metrics_top_1["Reader"]["exact_match"] == 1.0
    assert metrics_top_1["Reader"]["f1"] == 1.0
    assert metrics_top_1["Retriever"]["mrr"] == 0.5
    assert metrics_top_1["Retriever"]["map"] == 0.5
    assert metrics_top_1["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_1["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_1["Retriever"]["precision"] == 0.5

    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_retriever=2)
    
    assert metrics_top_2["Reader"]["exact_match"] == 1.0
    assert metrics_top_2["Reader"]["f1"] == 1.0
    assert metrics_top_2["Retriever"]["mrr"] == 0.5
    assert metrics_top_2["Retriever"]["map"] == 0.5
    assert metrics_top_2["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_2["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_2["Retriever"]["precision"] == 0.25

    metrics_top_3 = eval_result.calculate_metrics(simulated_top_k_retriever=3)
    
    assert metrics_top_3["Reader"]["exact_match"] == 1.0
    assert metrics_top_3["Reader"]["f1"] == 1.0
    assert metrics_top_3["Retriever"]["mrr"] == 0.5
    assert metrics_top_3["Retriever"]["map"] == 0.5
    assert metrics_top_3["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_3["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_3["Retriever"]["precision"] == 1.0/6
    

@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_simulated_top_k_reader_and_retriever(reader, retriever_with_docs):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 10}}
    )

    metrics_top_10 = eval_result.calculate_metrics(simulated_top_k_reader=1)

    assert metrics_top_10["Reader"]["exact_match"] == 0.5
    assert metrics_top_10["Reader"]["f1"] == 0.5
    assert metrics_top_10["Retriever"]["mrr"] == 0.5
    assert metrics_top_10["Retriever"]["map"] == 0.5
    assert metrics_top_10["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_10["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_10["Retriever"]["precision"] == 1.0/6

    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, simulated_top_k_retriever=1)
    
    assert metrics_top_1["Reader"]["exact_match"] == 0.5
    assert metrics_top_1["Reader"]["f1"] == 0.5
    assert metrics_top_1["Retriever"]["mrr"] == 0.5
    assert metrics_top_1["Retriever"]["map"] == 0.5
    assert metrics_top_1["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_1["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_1["Retriever"]["precision"] == 0.5

    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_reader=1, simulated_top_k_retriever=2)
    
    assert metrics_top_2["Reader"]["exact_match"] == 0.5
    assert metrics_top_2["Reader"]["f1"] == 0.5
    assert metrics_top_2["Retriever"]["mrr"] == 0.5
    assert metrics_top_2["Retriever"]["map"] == 0.5
    assert metrics_top_2["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_2["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_2["Retriever"]["precision"] == 0.25

    metrics_top_3 = eval_result.calculate_metrics(simulated_top_k_reader=1, simulated_top_k_retriever=3)
    
    assert metrics_top_3["Reader"]["exact_match"] == 0.5
    assert metrics_top_3["Reader"]["f1"] == 0.5
    assert metrics_top_3["Retriever"]["mrr"] == 0.5
    assert metrics_top_3["Retriever"]["map"] == 0.5
    assert metrics_top_3["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics_top_3["Retriever"]["recall_single_hit"] == 0.5
    assert metrics_top_3["Retriever"]["precision"] == 1.0/6
    

@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_wrong_examples(reader, retriever_with_docs):

    labels = [
        MultiLabel(labels=[Label(query="Who lives in Berlin?", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")]),
        MultiLabel(labels=[Label(query="Who lives in Munich?", answer=Answer(answer="Pete", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='something_else', content_type="text", content='My name is Pete and I live in Munich'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
    ]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=labels,
        params={"Retriever": {"top_k": 5}}, 
    )

    wrongs_retriever = eval_result.wrong_examples(node="Retriever", n=1)
    wrongs_reader = eval_result.wrong_examples(node="Reader", n=1)

    assert len(wrongs_retriever) == 1
    assert len(wrongs_reader) == 1


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_print_eval_report(reader, retriever_with_docs):

    labels = [
        MultiLabel(labels=[Label(query="Who lives in Berlin?", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")]),
        MultiLabel(labels=[Label(query="Who lives in Munich?", answer=Answer(answer="Pete", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='something_else', content_type="text", content='My name is Pete and I live in Munich'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
    ]

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=labels,
        params={"Retriever": {"top_k": 5}}, 
    )

    pipeline.print_eval_report(eval_result)


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_document_search_calculate_metrics(retriever_with_docs):
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert len(eval_result) == 1
    retriever_result = eval_result["Retriever"]
    retriever_berlin = retriever_result[retriever_result['query'] == "Who lives in Berlin?"]
    retriever_munich = retriever_result[retriever_result['query'] == "Who lives in Munich?"]

    assert retriever_berlin[retriever_berlin['rank'] == 1]["document_id"].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]["gold_document_ids"].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]["document_id"].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]["gold_document_ids"].iloc[0]
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_faq_calculate_metrics(retriever_with_docs):
    pipeline = FAQPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Docs2Answers" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6
    assert metrics["Docs2Answers"]["exact_match"] == 0.0
    assert metrics["Docs2Answers"]["f1"] == 0.0


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractive_qa_eval_translation(reader, retriever_with_docs, de_to_en_translator):
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    pipeline = TranslationWrapperPipeline(input_translator=de_to_en_translator, output_translator=de_to_en_translator, pipeline=pipeline)
    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Reader" in eval_result
    assert "OutputTranslator" in eval_result
    assert len(eval_result) == 3

    assert metrics["Reader"]["exact_match"] == 1.0
    assert metrics["Reader"]["f1"] == 1.0
    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6

    assert metrics["OutputTranslator"]["exact_match"] == 1.0
    assert metrics["OutputTranslator"]["f1"] == 1.0
    assert metrics["OutputTranslator"]["mrr"] == 0.5
    assert metrics["OutputTranslator"]["map"] == 0.5
    assert metrics["OutputTranslator"]["recall_multi_hit"] == 0.5
    assert metrics["OutputTranslator"]["recall_single_hit"] == 0.5
    assert metrics["OutputTranslator"]["precision"] == 1.0/6


@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_question_generation_eval(retriever_with_docs, question_generator):
    pipeline = RetrieverQuestionGenerationPipeline(retriever=retriever_with_docs, question_generator=question_generator)

    eval_result: EvaluationResult = pipeline.eval(
        labels=EVAL_LABELS,
        params={"Retriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "Retriever" in eval_result
    assert "Question Generator" in eval_result
    assert len(eval_result) == 2

    assert metrics["Retriever"]["mrr"] == 0.5
    assert metrics["Retriever"]["map"] == 0.5
    assert metrics["Retriever"]["recall_multi_hit"] == 0.5
    assert metrics["Retriever"]["recall_single_hit"] == 0.5
    assert metrics["Retriever"]["precision"] == 1.0/6

    assert metrics["Question Generator"]["mrr"] == 0.5
    assert metrics["Question Generator"]["map"] == 0.5
    assert metrics["Question Generator"]["recall_multi_hit"] == 0.5
    assert metrics["Question Generator"]["recall_single_hit"] == 0.5
    assert metrics["Question Generator"]["precision"] == 1.0/6


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_qa_multi_retriever_pipeline_eval(document_store_with_docs, reader):
    es_retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    dpr_retriever = DensePassageRetriever(document_store_with_docs)
    document_store_with_docs.update_embeddings(retriever=dpr_retriever)

    # QA Pipeline with two retrievers, we always want QA output
    pipeline = Pipeline()
    pipeline.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipeline.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
    pipeline.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "DPRRetriever"])

    # EVAL_QUERIES: 2 go dpr way
    # in Berlin goes es way
    labels = EVAL_LABELS + [
        MultiLabel(labels=[Label(query="in Berlin", answer=Answer(answer="Carla", offsets_in_context=[Span(11, 16)]), 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
            ]

    eval_result: EvaluationResult = pipeline.eval(
        labels=labels,
        params={"ESRetriever": {"top_k": 5}, "DPRRetriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "ESRetriever" in eval_result
    assert "DPRRetriever" in eval_result
    assert "QAReader" in eval_result
    assert len(eval_result) == 3

    assert metrics["DPRRetriever"]["mrr"] == 0.5
    assert metrics["DPRRetriever"]["map"] == 0.5
    assert metrics["DPRRetriever"]["recall_multi_hit"] == 0.5
    assert metrics["DPRRetriever"]["recall_single_hit"] == 0.5
    assert metrics["DPRRetriever"]["precision"] == 1.0/6

    assert metrics["ESRetriever"]["mrr"] == 1.0
    assert metrics["ESRetriever"]["map"] == 1.0
    assert metrics["ESRetriever"]["recall_multi_hit"] == 1.0
    assert metrics["ESRetriever"]["recall_single_hit"] == 1.0
    assert metrics["ESRetriever"]["precision"] == 1.0/3

    assert metrics["QAReader"]["exact_match"] == 1.0
    assert metrics["QAReader"]["f1"] == 1.0


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_multi_retriever_pipeline_eval(document_store_with_docs, reader):
    es_retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    dpr_retriever = DensePassageRetriever(document_store_with_docs)
    document_store_with_docs.update_embeddings(retriever=dpr_retriever)

    # QA Pipeline with two retrievers, no QA output
    pipeline = Pipeline()
    pipeline.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipeline.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])

    # EVAL_QUERIES: 2 go dpr way
    # in Berlin goes es way
    labels = EVAL_LABELS + [
        MultiLabel(labels=[Label(query="in Berlin", answer=None, 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
            ]

    eval_result: EvaluationResult = pipeline.eval(
        labels=labels,
        params={"ESRetriever": {"top_k": 5}, "DPRRetriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "ESRetriever" in eval_result
    assert "DPRRetriever" in eval_result
    assert len(eval_result) == 2

    assert metrics["DPRRetriever"]["mrr"] == 0.5
    assert metrics["DPRRetriever"]["map"] == 0.5
    assert metrics["DPRRetriever"]["recall_multi_hit"] == 0.5
    assert metrics["DPRRetriever"]["recall_single_hit"] == 0.5
    assert metrics["DPRRetriever"]["precision"] == 1.0/6

    assert metrics["ESRetriever"]["mrr"] == 1.0
    assert metrics["ESRetriever"]["map"] == 1.0
    assert metrics["ESRetriever"]["recall_multi_hit"] == 1.0
    assert metrics["ESRetriever"]["recall_single_hit"] == 1.0
    assert metrics["ESRetriever"]["precision"] == 1.0/3


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("reader", ["farm"], indirect=True)
def test_multi_retriever_pipeline_with_asymmetric_qa_eval(document_store_with_docs, reader):
    es_retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    dpr_retriever = DensePassageRetriever(document_store_with_docs)
    document_store_with_docs.update_embeddings(retriever=dpr_retriever)

    # QA Pipeline with two retrievers, we only get QA output from dpr
    pipeline = Pipeline()
    pipeline.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    pipeline.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
    pipeline.add_node(component=reader, name="QAReader", inputs=["DPRRetriever"])

    # EVAL_QUERIES: 2 go dpr way
    # in Berlin goes es way
    labels = EVAL_LABELS + [
        MultiLabel(labels=[Label(query="in Berlin", answer=None, 
            document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type="text", content='My name is Carla and I live in Berlin'), 
            is_correct_answer=True, is_correct_document=True, origin="gold-label")])
            ]

    eval_result: EvaluationResult = pipeline.eval(
        labels=labels,
        params={"ESRetriever": {"top_k": 5}, "DPRRetriever": {"top_k": 5}}
    )

    metrics = eval_result.calculate_metrics()

    assert "ESRetriever" in eval_result
    assert "DPRRetriever" in eval_result
    assert "DPRRetriever" in eval_result
    assert "QAReader" in eval_result
    assert len(eval_result) == 3

    assert metrics["DPRRetriever"]["mrr"] == 0.5
    assert metrics["DPRRetriever"]["map"] == 0.5
    assert metrics["DPRRetriever"]["recall_multi_hit"] == 0.5
    assert metrics["DPRRetriever"]["recall_single_hit"] == 0.5
    assert metrics["DPRRetriever"]["precision"] == 1.0/6

    assert metrics["ESRetriever"]["mrr"] == 1.0
    assert metrics["ESRetriever"]["map"] == 1.0
    assert metrics["ESRetriever"]["recall_multi_hit"] == 1.0
    assert metrics["ESRetriever"]["recall_single_hit"] == 1.0
    assert metrics["ESRetriever"]["precision"] == 1.0/3

    assert metrics["QAReader"]["exact_match"] == 1.0
    assert metrics["QAReader"]["f1"] == 1.0
