import json

from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document, GeneratedAnswer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.evaluation.eval import eval
from haystack.evaluation.metrics import Metric


def test_bm25_rag_pipeline(tmp_path):
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    rag_pipeline.get_component("retriever").document_store.write_documents(documents)

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
        for question in questions
    ]

    expected_outputs = [
        {
            "answer_builder": {
                "answers": [
                    GeneratedAnswer(
                        data="Jean",
                        query="Who lives in Paris?",
                        documents=[
                            Document(content="My name is Jean and I live in Paris.", score=0.33144005810482535),
                            Document(content="My name is Giorgio and I live in Rome.", score=-0.17938556566116537),
                            Document(content="My name is Mark and I live in Berlin.", score=-0.17938556566116537),
                        ],
                        meta={},
                    )
                ]
            }
        },
        {
            "answer_builder": {
                "answers": [
                    GeneratedAnswer(
                        data="Mark",
                        query="Who lives in Berlin?",
                        documents=[
                            Document(content="My name is Mark and I live in Berlin.", score=0.33144005810482535),
                            Document(content="My name is Giorgio and I live in Rome.", score=-0.17938556566116537),
                            Document(content="My name is Jean and I live in Paris.", score=-0.17938556566116537),
                        ],
                        meta={},
                    )
                ]
            }
        },
        {
            "answer_builder": {
                "answers": [
                    GeneratedAnswer(
                        data="Giorgio",
                        query="Who lives in Rome?",
                        documents=[
                            Document(content="My name is Giorgio and I live in Rome.", score=0.33144005810482535),
                            Document(content="My name is Mark and I live in Berlin.", score=-0.17938556566116537),
                            Document(content="My name is Jean and I live in Paris.", score=-0.17938556566116537),
                        ],
                        meta={},
                    )
                ]
            }
        },
    ]

    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
    assert eval_result.runnable.to_dict() == rag_pipeline.to_dict()

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


def test_embedding_retrieval_rag_pipeline(tmp_path):
    # Create the RAG pipeline
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="text_embedder"
    )
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever"
    )
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    document_store = rag_pipeline.get_component("retriever").document_store
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        name="document_embedder",
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="document_writer")
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})

    # Query and assert
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [
        {
            "prompt_builder": {"question": question},
            "text_embedder": {"text": question},
            "answer_builder": {"query": question},
        }
        for question in questions
    ]

    expected_outputs = [
        {
            "answer_builder": {
                "answers": [
                    GeneratedAnswer(
                        data="Jean",
                        query="Who lives in Paris?",
                        documents=[
                            Document(content="My name is Jean and I live in Paris.", score=0.33144005810482535),
                            Document(content="My name is Giorgio and I live in Rome.", score=-0.17938556566116537),
                            Document(content="My name is Mark and I live in Berlin.", score=-0.17938556566116537),
                        ],
                        meta={},
                    )
                ]
            }
        },
        {
            "answer_builder": {
                "answers": [
                    GeneratedAnswer(
                        data="Mark",
                        query="Who lives in Berlin?",
                        documents=[
                            Document(content="My name is Mark and I live in Berlin.", score=0.33144005810482535),
                            Document(content="My name is Giorgio and I live in Rome.", score=-0.17938556566116537),
                            Document(content="My name is Jean and I live in Paris.", score=-0.17938556566116537),
                        ],
                        meta={},
                    )
                ]
            }
        },
        {
            "answer_builder": {
                "answers": [
                    GeneratedAnswer(
                        data="Giorgio",
                        query="Who lives in Rome?",
                        documents=[
                            Document(content="My name is Giorgio and I live in Rome.", score=0.33144005810482535),
                            Document(content="My name is Mark and I live in Berlin.", score=-0.17938556566116537),
                            Document(content="My name is Jean and I live in Paris.", score=-0.17938556566116537),
                        ],
                        meta={},
                    )
                ]
            }
        },
    ]

    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
    assert eval_result.runnable.to_dict() == rag_pipeline.to_dict()

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
