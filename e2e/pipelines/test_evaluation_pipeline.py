# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import pytest

from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
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

EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def indexing_pipeline(documents: List[Document]):
    """Indexing the documents"""
    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDINGS_MODEL, progress_bar=False)
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")  # type: ignore
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")  # type: ignore
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})
    return document_store


def rag_pipeline(document_store: InMemoryDocumentStore, top_k: int):  # type: ignore
    """RAG pipeline"""
    template = """
        You have to answer the following question based on the given context information only.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
    rag = Pipeline()
    rag.add_component("embedder", SentenceTransformersTextEmbedder(model=EMBEDDINGS_MODEL, progress_bar=False))  # type: ignore
    rag.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=top_k))  # type: ignore
    rag.add_component("prompt_builder", PromptBuilder(template=template))  # type: ignore
    rag.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))  # type: ignore
    rag.add_component("answer_builder", AnswerBuilder())  # type: ignore
    rag.connect("embedder", "retriever.query_embedding")
    rag.connect("retriever", "prompt_builder.documents")
    rag.connect("prompt_builder", "generator")
    rag.connect("generator.replies", "answer_builder.replies")
    rag.connect("generator.meta", "answer_builder.meta")
    rag.connect("retriever", "answer_builder.documents")

    return rag


def evaluation_pipeline():
    """
    Create an evaluation pipeline with the following evaluators:

    - DocumentMRREvaluator
    - FaithfulnessEvaluator
    - SASEvaluator
    - DocumentMAPEvaluator
    - DocumentRecallEvaluator
    - ContextRelevanceEvaluator
    """
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
    eval_pipeline.add_component("groundedness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas", SASEvaluator(model=EMBEDDINGS_MODEL))
    eval_pipeline.add_component("doc_map", DocumentMAPEvaluator())
    eval_pipeline.add_component("doc_recall_single_hit", DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT))
    eval_pipeline.add_component("doc_recall_multi_hit", DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT))
    eval_pipeline.add_component("relevance", ContextRelevanceEvaluator())

    return eval_pipeline


def built_eval_input(questions, truth_docs, truth_answers, retrieved_docs, contexts, pred_answers):
    """Helper function to build the input for the evaluation pipeline"""
    return {
        "doc_mrr": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
        "groundedness": {"questions": questions, "contexts": contexts, "predicted_answers": pred_answers},
        "sas": {"predicted_answers": pred_answers, "ground_truth_answers": truth_answers},
        "doc_map": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
        "doc_recall_single_hit": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
        "doc_recall_multi_hit": {"ground_truth_documents": truth_docs, "retrieved_documents": retrieved_docs},
        "relevance": {"questions": questions, "contexts": contexts},
    }


def run_rag_pipeline(documents, evaluation_questions, rag_pipeline_a):
    """
    Run the RAG pipeline and return the contexts, predicted answers, retrieved documents and ground truth documents
    """

    truth_docs = []
    retrieved_docs = []
    contexts = []
    predicted_answers = []

    for q in evaluation_questions:
        response = rag_pipeline_a.run(
            {
                "embedder": {"text": q["question"]},
                "prompt_builder": {"question": q["question"]},
                "answer_builder": {"query": q["question"]},
            }
        )
        truth_docs.append([doc for doc in documents if doc.meta["name"] in q["ground_truth_doc"] and doc.content])
        retrieved_docs.append(response["answer_builder"]["answers"][0].documents)
        contexts.append([doc.content for doc in response["answer_builder"]["answers"][0].documents])
        predicted_answers.append(response["answer_builder"]["answers"][0].data)

    return contexts, predicted_answers, retrieved_docs, truth_docs


def built_input_for_results_eval(rag_results):
    """Helper function to build the input for the results evaluation"""
    return {
        "Mean Reciprocal Rank": {
            "individual_scores": rag_results["doc_mrr"]["individual_scores"],
            "score": rag_results["doc_mrr"]["score"],
        },
        "Semantic Answer Similarity": {
            "individual_scores": rag_results["sas"]["individual_scores"],
            "score": rag_results["sas"]["score"],
        },
        "Faithfulness": {
            "individual_scores": rag_results["groundedness"]["individual_scores"],
            "score": rag_results["groundedness"]["score"],
        },
        "Document MAP": {
            "individual_scores": rag_results["doc_map"]["individual_scores"],
            "score": rag_results["doc_map"]["score"],
        },
        "Document Recall Single Hit": {
            "individual_scores": rag_results["doc_recall_single_hit"]["individual_scores"],
            "score": rag_results["doc_recall_single_hit"]["score"],
        },
        "Document Recall Multi Hit": {
            "individual_scores": rag_results["doc_recall_multi_hit"]["individual_scores"],
            "score": rag_results["doc_recall_multi_hit"]["score"],
        },
        "Contextual Relevance": {
            "individual_scores": rag_results["relevance"]["individual_scores"],
            "score": rag_results["relevance"]["score"],
        },
    }


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_evaluation_pipeline(samples_path):
    """Test an evaluation pipeline"""
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
        {
            "question": "What is materialism?",
            "answer": "a form of philosophical monism",
            "ground_truth_doc": ["Materialism.txt"],
        },
    ]

    questions = [q["question"] for q in eval_questions]
    truth_answers = [q["answer"] for q in eval_questions]

    # indexing documents
    docs = []
    full_path = os.path.join(str(samples_path) + "/test_documents/")
    for article in os.listdir(full_path):
        with open(f"{full_path}/{article}", "r") as f:
            for text in f.read().split("\n"):
                if doc := Document(content=text, meta={"name": article}) if text else None:
                    docs.append(doc)
    doc_store = indexing_pipeline(docs)

    # running the RAG pipeline A + evaluation pipeline
    rag_pipeline_a = rag_pipeline(doc_store, top_k=2)
    contexts_a, pred_answers_a, retrieved_docs_a, truth_docs = run_rag_pipeline(docs, eval_questions, rag_pipeline_a)
    eval_pipeline = evaluation_pipeline()
    eval_input = built_eval_input(questions, truth_docs, truth_answers, retrieved_docs_a, contexts_a, pred_answers_a)
    results_rag_a = eval_pipeline.run(eval_input)

    # running the evaluation EvaluationRunResult
    inputs_a = {
        "question": questions,
        "contexts": contexts_a,
        "answer": truth_answers,
        "predicted_answer": pred_answers_a,
    }
    results_a = built_input_for_results_eval(results_rag_a)
    evaluation_result_a = EvaluationRunResult(run_name="rag_pipeline_a", results=results_a, inputs=inputs_a)
    df_score_report = evaluation_result_a.score_report()

    # assert the score report has all the metrics
    assert len(df_score_report) == 7
    assert list(df_score_report.columns) == ["metrics", "score"]
    assert list(df_score_report.metrics) == [
        "Mean Reciprocal Rank",
        "Semantic Answer Similarity",
        "Faithfulness",
        "Document MAP",
        "Document Recall Single Hit",
        "Document Recall Multi Hit",
        "Contextual Relevance",
    ]

    # assert the evaluation result has all the metrics, inputs and questions
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
        "Contextual Relevance",
    ]
    assert len(df) == 3

    # running the RAG pipeline B
    rag_pipeline_b = rag_pipeline(doc_store, top_k=4)
    contexts_b, pred_answers_b, retrieved_docs_b, truth_docs = run_rag_pipeline(docs, eval_questions, rag_pipeline_b)
    eval_input = built_eval_input(questions, truth_docs, truth_answers, retrieved_docs_b, contexts_b, pred_answers_b)
    results_rag_b = eval_pipeline.run(eval_input)

    inputs_b = {
        "question": questions,
        "contexts": contexts_b,
        "answer": truth_answers,
        "predicted_answer": pred_answers_b,
    }
    results_b = built_input_for_results_eval(results_rag_b)
    evaluation_result_b = EvaluationRunResult(run_name="rag_pipeline_b", results=results_b, inputs=inputs_b)
    df_comparative = evaluation_result_a.comparative_individual_scores_report(evaluation_result_b)

    # assert the comparative score report has all the metrics, inputs and questions
    assert len(df_comparative) == 3
    assert list(df_comparative.columns) == [
        "question",
        "contexts",
        "answer",
        "predicted_answer",
        "rag_pipeline_a_Mean Reciprocal Rank",
        "rag_pipeline_a_Semantic Answer Similarity",
        "rag_pipeline_a_Faithfulness",
        "rag_pipeline_a_Document MAP",
        "rag_pipeline_a_Document Recall Single Hit",
        "rag_pipeline_a_Document Recall Multi Hit",
        "rag_pipeline_a_Contextual Relevance",
        "rag_pipeline_b_Mean Reciprocal Rank",
        "rag_pipeline_b_Semantic Answer Similarity",
        "rag_pipeline_b_Faithfulness",
        "rag_pipeline_b_Document MAP",
        "rag_pipeline_b_Document Recall Single Hit",
        "rag_pipeline_b_Document Recall Multi Hit",
        "rag_pipeline_b_Contextual Relevance",
    ]
