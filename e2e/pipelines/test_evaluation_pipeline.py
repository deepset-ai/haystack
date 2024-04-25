import os
from typing import List

import pytest

from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.evaluators import (
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
    FaithfulnessEvaluator,
    SASEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.evaluation import EvaluationRunResult

embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"


def indexing_pipeline(documents: List[Document]):
    """Indexing the documents"""
    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embeddings_model)
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})
    return document_store


def rag_pipeline(document_store: InMemoryDocumentStore, top_k: int):
    """Building the RAG pipeline"""
    template = """
        You have to answer the following question based on the given context information only.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=embeddings_model))
    rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=top_k))
    rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    rag_pipeline.add_component("generator", OpenAIGenerator(model="gpt-3.5-turbo"))
    rag_pipeline.add_component("answer_builder", AnswerBuilder())
    rag_pipeline.connect("embedder", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "generator")
    rag_pipeline.connect("generator.replies", "answer_builder.replies")
    rag_pipeline.connect("generator.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    return rag_pipeline


def evaluation_pipeline(questions, truth_docs, truth_answers, retrieved_docs, contexts, pred_answers):
    """
    Run the evaluation pipeline
    """
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
    eval_pipeline.add_component("groundness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas", SASEvaluator(model=embeddings_model))
    eval_pipeline.add_component("doc_map", DocumentMAPEvaluator())
    eval_pipeline.add_component("doc_recall_single_hit", DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT))
    eval_pipeline.add_component("doc_recall_multi_hit", DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT))

    return eval_pipeline.run(
        {
            "doc_mrr": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
            "groundness": {"questions": questions, "contexts": contexts, "responses": truth_answers},
            "sas": {"predicted_answers": pred_answers, "ground_truth_answers": truth_answers},
            "doc_map": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
            "doc_recall_single_hit": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
            "doc_recall_multi_hit": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
        }
    )


def run_rag_pipeline(documents, evaluation_questions, rag_pipeline_a):
    """
    Run the RAG pipeline and return the contexts, predicted answers, retrieved documents and ground truth documents
    """

    truth_docs = []
    retrieved_docs = []
    contexts = []
    pred_answers = []

    for q in evaluation_questions:
        response = rag_pipeline_a.run(
            {
                "embedder": {"text": q["question"]},
                "prompt_builder": {"question": q["question"]},
                "answer_builder": {"query": q["question"]},
            }
        )
        truth_docs.append([doc for doc in documents if doc.meta["name"] in q["ground_truth_doc"]])
        retrieved_docs.append(response["answer_builder"]["answers"][0].documents)
        contexts.append([doc.content for doc in response["answer_builder"]["answers"][0].documents])
        pred_answers.append(response["answer_builder"]["answers"][0].data)

    return contexts, pred_answers, retrieved_docs, truth_docs


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_evaluation_pipeline(samples_path):
    """
    Test custom built evaluation pipeline
    """

    eval_questions = [
        {
            "question": 'What falls within the term "cultural anthropology"?',
            "answer": "the ideology and analytical stance of cultural relativism",
            "ground_truth_doc": ["Culture.txt"],
        },
        {
            "question": "Who was the spiritual guide during the Protestant Reformation?",
            "answer": "Martin Bucer",
            "ground_truth_doc": ["Strasbourg.txt"],
        },
        {"question": "What separates many annelids' segments?", "answer": "Septa", "ground_truth_doc": ["Annelid.txt"]},
        {
            "question": "What is materialism?",
            "answer": "a form of philosophical monism",
            "ground_truth_doc": ["Materialism.txt"],
        },
    ]

    docs = []
    full_path = os.path.join(str(samples_path) + "/test_documents/")
    for article in os.listdir(full_path):
        with open(f"{full_path}/{article}", "r") as f:
            for text in f.read().split("\n"):
                docs.append(Document(content=text, meta={"name": article}))
    doc_store = indexing_pipeline(docs)

    questions = [q["question"] for q in eval_questions]
    truth_answers = [q["answer"] for q in eval_questions]

    rag_pipeline_a = rag_pipeline(doc_store, top_k=3)
    contexts_a, pred_answers_a, retrieved_docs_a, truth_docs = run_rag_pipeline(docs, eval_questions, rag_pipeline_a)
    results_rag_a = evaluation_pipeline(
        questions, truth_docs, truth_answers, retrieved_docs_a, contexts_a, pred_answers_a
    )

    inputs_a = {
        "question": questions,
        "contexts": contexts_a,
        "answer": truth_answers,
        "predicted_answer": pred_answers_a,
    }
    results_a = {
        "Mean Reciprocal Rank": {
            "individual_scores": results_rag_a["doc_mrr"]["individual_scores"],
            "score": results_rag_a["doc_mrr"]["score"],
        },
        "Semantic Answer Similarity": {
            "individual_scores": results_rag_a["sas"]["individual_scores"],
            "score": results_rag_a["sas"]["score"],
        },
        "Faithfulness": {
            "individual_scores": results_rag_a["groundness"]["individual_scores"],
            "score": results_rag_a["groundness"]["score"],
        },
        "Document MAP": {
            "individual_scores": results_rag_a["doc_map"]["individual_scores"],
            "score": results_rag_a["doc_map"]["score"],
        },
        "Document Recall Single Hit": {
            "individual_scores": results_rag_a["doc_recall_single_hit"]["individual_scores"],
            "score": results_rag_a["doc_recall_single_hit"]["score"],
        },
        "Document Recall Multi Hit": {
            "individual_scores": results_rag_a["doc_recall_multi_hit"]["individual_scores"],
            "score": results_rag_a["doc_recall_multi_hit"]["score"],
        },
    }
    evaluation_result_a = EvaluationRunResult(run_name="rag_pipeline_a", results=results_a, inputs=inputs_a)
    df_score_report = evaluation_result_a.score_report()

    assert len(df_score_report) == 6
    assert list(df_score_report.columns) == ["score"]
    assert list(df_score_report.index) == [
        "Mean Reciprocal Rank",
        "Semantic Answer Similarity",
        "Faithfulness",
        "Document MAP",
        "Document Recall Single Hit",
        "Document Recall Multi Hit",
    ]
    df = evaluation_result_a.to_pandas()
    assert list(df.columns) == [
        "question",
        "contexts",
        "answer",
        "predicted_answer",
        "Mean Reciprocal Rank",
        "Semantic Answer Similarity",
        "Faithfulness",
        "Document MAP",
        "Document Recall Single Hit",
        "Document Recall Multi Hit",
    ]
    assert len(df) == 4

    rag_pipeline_b = rag_pipeline(doc_store, top_k=5)
    contexts_b, pred_answers_b, retrieved_docs_b, truth_docs = run_rag_pipeline(docs, eval_questions, rag_pipeline_b)
    results_rag_b = evaluation_pipeline(
        questions, truth_docs, truth_answers, retrieved_docs_b, contexts_b, pred_answers_b
    )

    inputs_b = {
        "question": questions,
        "contexts": contexts_a,
        "answer": truth_answers,
        "predicted_answer": pred_answers_b,
    }
    results_b = {
        "Mean Reciprocal Rank": {
            "individual_scores": results_rag_b["doc_mrr"]["individual_scores"],
            "score": results_rag_b["doc_mrr"]["score"],
        },
        "Semantic Answer Similarity": {
            "individual_scores": results_rag_b["sas"]["individual_scores"],
            "score": results_rag_b["sas"]["score"],
        },
        "Faithfulness": {
            "individual_scores": results_rag_b["groundness"]["individual_scores"],
            "score": results_rag_b["groundness"]["score"],
        },
        "Document MAP": {
            "individual_scores": results_rag_b["doc_map"]["individual_scores"],
            "score": results_rag_b["doc_map"]["score"],
        },
        "Document Recall Single Hit": {
            "individual_scores": results_rag_b["doc_recall_single_hit"]["individual_scores"],
            "score": results_rag_b["doc_recall_single_hit"]["score"],
        },
        "Document Recall Multi Hit": {
            "individual_scores": results_rag_b["doc_recall_multi_hit"]["individual_scores"],
            "score": results_rag_b["doc_recall_multi_hit"]["score"],
        },
    }
    evaluation_result_b = EvaluationRunResult(run_name="rag_pipeline_b", results=results_b, inputs=inputs_b)
    df_comparative = evaluation_result_a.comparative_individual_scores_report(evaluation_result_b)

    print(df_comparative)
