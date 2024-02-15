import json
import pytest

from haystack import Pipeline
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document, ExtractedAnswer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.evaluation.eval import eval
from haystack.evaluation.metrics import Metric


def test_extractive_qa_pipeline(tmp_path):
    # Create the pipeline
    qa_pipeline = Pipeline()
    qa_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    qa_pipeline.add_component(instance=ExtractiveReader(model="deepset/tinyroberta-squad2"), name="reader")
    qa_pipeline.connect("retriever", "reader")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    qa_pipeline.get_component("retriever").document_store.write_documents(documents)

    # Query and assert
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [{"retriever": {"query": question}, "reader": {"query": question, "top_k": 1}} for question in questions]
    expected_outputs = [
        {
            "reader": {
                "answers": [
                    ExtractedAnswer(
                        query="Who lives in Paris?",
                        score=0.7713339924812317,
                        data="Jean and I",
                        document=Document(content="My name is Jean and I live in Paris.", score=0.33144005810482535),
                        context=None,
                        document_offset=ExtractedAnswer.Span(start=11, end=21),
                        context_offset=None,
                        meta={},
                    ),
                    ExtractedAnswer(
                        query="Who lives in Paris?",
                        score=0.2286660075187683,
                        data=None,
                        document=None,
                        context=None,
                        document_offset=None,
                        context_offset=None,
                        meta={},
                    ),
                ]
            }
        },
        {
            "reader": {
                "answers": [
                    ExtractedAnswer(
                        query="Who lives in Berlin?",
                        score=0.7047999501228333,
                        data="Mark and I",
                        document=Document(content="My name is Mark and I live in Berlin.", score=0.33144005810482535),
                        context=None,
                        document_offset=ExtractedAnswer.Span(start=11, end=21),
                        context_offset=None,
                        meta={},
                    ),
                    ExtractedAnswer(
                        query="Who lives in Berlin?",
                        score=0.29520004987716675,
                        data=None,
                        document=None,
                        context=None,
                        document_offset=None,
                        context_offset=None,
                        meta={},
                    ),
                ]
            }
        },
        {
            "reader": {
                "answers": [
                    ExtractedAnswer(
                        query="Who lives in Rome?",
                        score=0.7661304473876953,
                        data="Giorgio and I",
                        document=Document(content="My name is Giorgio and I live in Rome.", score=0.33144005810482535),
                        context=None,
                        document_offset=ExtractedAnswer.Span(start=11, end=24),
                        context_offset=None,
                        meta={},
                    ),
                    ExtractedAnswer(
                        query="Who lives in Rome?",
                        score=0.2338695526123047,
                        data=None,
                        document=None,
                        context=None,
                        document_offset=None,
                        context_offset=None,
                        meta={},
                    ),
                ]
            }
        },
    ]

    eval_result = eval(qa_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
    assert eval_result.runnable.to_dict() == qa_pipeline.to_dict()

    # Test Exact Match
    em_default = eval_result.calculate_metrics(Metric.EM, output_key="answers")
    em_custom_parameters = eval_result.calculate_metrics(
        Metric.EM, output_key="answers", ignore_case=True, ignore_punctuation=True, ignore_numbers=True
    )
    # Save EM metric results to json
    em_default.save(tmp_path / "exact_match_score.json")

    assert em_default["exact_match"] == 1.0
    assert em_custom_parameters["exact_match"] == 1.0
    with open(tmp_path / "exact_match_score.json", "r") as f:
        assert em_default == json.load(f)

    # Test F1
    f1_default = eval_result.calculate_metrics(Metric.F1, output_key="answers")
    f1_custom_parameters = eval_result.calculate_metrics(
        Metric.F1, output_key="answers", ignore_case=True, ignore_punctuation=True, ignore_numbers=True
    )
    # Save F1 metric results to json
    f1_default.save(tmp_path / "f1_score.json")

    assert f1_default["f1"] == 1.0
    assert f1_custom_parameters["f1"] == 1.0
    with open(tmp_path / "f1_score.json", "r") as f:
        assert f1_default == json.load(f)

    # Test SAS
    sas_default = eval_result.calculate_metrics(
        Metric.SAS, output_key="answers", model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    sas_custom_parameters = eval_result.calculate_metrics(
        Metric.SAS,
        output_key="answers",
        ignore_case=True,
        ignore_punctuation=True,
        ignore_numbers=True,
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    # Save SAS metric results to json
    sas_default.save(tmp_path / "sas_score.json")

    assert sas_default["sas"] == pytest.approx(1.0)
    assert sas_default["scores"] == pytest.approx([1.0, 1.0, 1.0])
    assert sas_custom_parameters["sas"] == pytest.approx(0.9996823, abs=1e-5)
    assert sas_custom_parameters["scores"] == pytest.approx([0.999672, 0.999608, 0.999767])

    with open(tmp_path / "sas_score.json", "r") as f:
        assert sas_default == json.load(f)
