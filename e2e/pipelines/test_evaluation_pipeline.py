import os
from typing import List

from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.evaluators import (
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
    EvaluationResult,
    FaithfulnessEvaluator,
    SASEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


def indexing(documents: List[Document]):
    """
    Indexing the documents
    """
    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})
    return document_store


def build_rag_pipeline(document_store, top_k=2):
    """
    Building the RAG pipeline
    """
    template = """
        You have to answer the following question based on the given context information only.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

    rag_pipeline_1 = Pipeline()
    rag_pipeline_1.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    rag_pipeline_1.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=top_k))
    rag_pipeline_1.add_component("prompt_builder", PromptBuilder(template=template))
    rag_pipeline_1.add_component("generator", OpenAIGenerator(model="gpt-3.5-turbo"))
    rag_pipeline_1.add_component("answer_builder", AnswerBuilder())

    rag_pipeline_1.connect("query_embedder", "retriever.query_embedding")
    rag_pipeline_1.connect("retriever", "prompt_builder.documents")
    rag_pipeline_1.connect("prompt_builder", "generator")
    rag_pipeline_1.connect("generator.replies", "answer_builder.replies")
    rag_pipeline_1.connect("generator.meta", "answer_builder.meta")
    rag_pipeline_1.connect("retriever", "answer_builder.documents")

    return rag_pipeline_1


def test_evaluation_pipeline(samples_path):
    """
    Test the evaluation pipeline
    """
    documents = []

    def create_document(text: str, name: str):
        return Document(content=text, meta={"name": name})

    for root, dirs, files in os.walk(str(samples_path) + "/test_documents/"):
        for article in files:
            with open(f"{root}/{article}", "r") as f:
                raw_texts = f.read().split("\n")
                for text in raw_texts:
                    documents.append(create_document(text, article))

    document_store = indexing(documents)

    # collect all the data for evaluation
    all_questions = []
    all_ground_truth_documents = []
    all_ground_truth_answers = []
    all_retrieved_documents = []
    all_contexts = []
    all_answers = []

    questions = [
        {
            "question": "Who re-translated the Reflections into French?",
            "answer": ["Louis XVI"],
            "ground_truth_doc": ["Edmund_Burke.txt"],
        },
        {
            "question": "What was Kerry's role in the Yale Political Union as a junior?",
            "answer": ["President of the Union"],
            "ground_truth_doc": ["John_Kerry.txt"],
        },
        {
            "question": 'What falls within the term "cultural anthropology"?',
            "answer": ["the ideology and analytical stance of cultural relativism"],
            "ground_truth_doc": ["Culture.txt"],
        },
        {
            "question": "Who was the spiritual guide during the Protestant Reformation?",
            "answer": ["Martin Bucer"],
            "ground_truth_doc": ["Strasbourg.txt"],
        },
        {
            "question": "What separates many annelids' segments?",
            "answer": ["Septa"],
            "ground_truth_doc": ["Annelid.txt"],
        },
        {
            "question": "What is materialism?",
            "answer": ["a form of philosophical monism"],
            "ground_truth_doc": ["Materialism.txt"],
        },
        {
            "question": "Who did the Hungarian nobility elect as King of Hungary?",
            "answer": ["Matthias"],
            "ground_truth_doc": ["Late_Middle_Ages.txt"],
        },
    ]

    rag_pipeline_1 = build_rag_pipeline(document_store, top_k=2)

    # ToDo: do this in batch to avoid multiple calls to the pipeline
    for q in questions:
        question = q["question"]
        answer = q["answer"]
        ground_truth_docs = [doc for doc in documents if doc.meta["name"] in q["ground_truth_doc"]]
        all_ground_truth_documents.append(ground_truth_docs)
        all_ground_truth_answers.append(answer[0])
        all_questions.append(question)

        response = rag_pipeline_1.run(
            {
                "query_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        all_retrieved_documents.append(response["answer_builder"]["answers"][0].documents)
        all_contexts.append([doc.content for doc in response["answer_builder"]["answers"][0].documents])
        all_answers.append(response["answer_builder"]["answers"][0].data)

    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
    eval_pipeline.add_component("groundness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2"))
    eval_pipeline.add_component("doc_map", DocumentMAPEvaluator())
    eval_pipeline.add_component("doc_recall_single_hit", DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT))
    eval_pipeline.add_component("doc_recall_multi_hit", DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT))

    results = eval_pipeline.run(
        {
            "doc_mrr": {
                "ground_truth_documents": all_ground_truth_documents,
                "retrieved_documents": all_retrieved_documents,
            },
            "groundness": {"questions": all_questions, "contexts": all_contexts, "responses": all_answers},
            "sas": {"predicted_answers": all_answers, "ground_truth_answers": all_ground_truth_answers},
            "doc_map": {
                "ground_truth_documents": all_ground_truth_documents,
                "retrieved_documents": all_retrieved_documents,
            },
            "doc_recall_single_hit": {
                "ground_truth_documents": all_ground_truth_documents,
                "retrieved_documents": all_retrieved_documents,
            },
            "doc_recall_multi_hit": {
                "ground_truth_documents": all_ground_truth_documents,
                "retrieved_documents": all_retrieved_documents,
            },
        }
    )

    data = {
        "inputs": {
            "question": all_questions,
            "contexts": all_contexts,
            "answer": all_ground_truth_answers,
            "predicted_answer": all_answers,
        },
        "metrics": [
            {
                "name": "Mean Reciprocal Rank",
                "individual_scores": results["doc_mrr"]["individual_scores"],
                "score": results["doc_mrr"]["score"],
            },
            {
                "name": "Semantic Answer Similarity",
                "individual_scores": results["sas"]["individual_scores"],
                "score": results["sas"]["score"],
            },
            {
                "name": "Faithfulness",
                "individual_scores": results["groundness"]["individual_scores"],
                "score": results["groundness"]["score"],
            },
            {
                "name": "Document MAP",
                "individual_scores": results["doc_map"]["individual_scores"],
                "score": results["doc_map"]["score"],
            },
            {
                "name": "Document Recall Single Hit",
                "individual_scores": results["doc_recall_single_hit"]["individual_scores"],
                "score": results["doc_recall_single_hit"]["score"],
            },
            {
                "name": "Document Recall Multi Hit",
                "individual_scores": results["doc_recall_multi_hit"]["individual_scores"],
                "score": results["doc_recall_multi_hit"]["score"],
            },
        ],
    }

    evaluation_result = EvaluationResult(pipeline_name="pipe_1", results=data)

    print(evaluation_result)
