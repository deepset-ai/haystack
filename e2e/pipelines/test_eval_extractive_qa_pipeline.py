import json

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
    qa_pipeline.add_component(instance=ExtractiveReader(model_name_or_path="deepset/tinyroberta-squad2"), name="reader")
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
                        document=Document(
                            id="6c90b78ad94e4e634e2a067b5fe2d26d4ce95405ec222cbaefaeb09ab4dce81e",
                            content="My name is Jean and I live in Paris.",
                            score=0.33144005810482535,
                        ),
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
                        document=Document(
                            id="10a183e965c2e107e20507c717f16559c58a8ba4bc7c577ea8dc32a8d6ca7a20",
                            content="My name is Mark and I live in Berlin.",
                            score=0.33144005810482535,
                        ),
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
                        document=Document(
                            id="fb0f1efe94b3c78aa1c4e5a17a5ef8270f70e89d36a3665c8362675e8a769a27",
                            content="My name is Giorgio and I live in Rome.",
                            score=0.33144005810482535,
                        ),
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

    metrics = eval_result.calculate_metrics(Metric.EM)
    # Save metric results to json
    metrics.save(tmp_path / "exact_match_score.json")

    assert metrics["exact_match"] == 1.0
    with open(tmp_path / "exact_match_score.json", "r") as f:
        assert metrics == json.load(f)
