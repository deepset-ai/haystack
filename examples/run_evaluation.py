import json
import os
import random
from typing import List

from haystack_integrations.components.evaluators.ragas import RagasEvaluator, RagasMetric

from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.evaluators import AnswerExactMatchEvaluator
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


def load_transformed_squad():
    with open("transformed_squad/questions.jsonl", "r") as f:
        questions = [json.loads(x) for x in f.readlines()]

    def create_document(text: str, name: str):
        return Document(content=text, meta={"name": name})

    # walk through the files in the directory and transform each text file into a Document
    documents = []
    for root, dirs, files in os.walk("transformed_squad/articles/"):
        for article in files:
            with open(f"{root}/{article}", "r") as f:
                raw_texts = f.read().split("\n")
                for text in raw_texts:
                    documents.append(create_document(text, article.replace(".txt", "")))

    return questions, documents


def indexing(documents: List[Document]):
    document_store = InMemoryDocumentStore()

    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")

    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store


def run_evaluation():
    template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

    questions, documents = load_transformed_squad()
    document_store = indexing(documents)

    rag_pipeline_1 = Pipeline()
    rag_pipeline_1.add_component(
        "query_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    rag_pipeline_1.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=2))
    rag_pipeline_1.add_component("prompt_builder", PromptBuilder(template=template))
    rag_pipeline_1.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    rag_pipeline_1.add_component("answer_builder", AnswerBuilder())

    rag_pipeline_1.connect("query_embedder", "retriever.query_embedding")
    rag_pipeline_1.connect("retriever", "prompt_builder.documents")
    rag_pipeline_1.connect("prompt_builder", "llm")
    rag_pipeline_1.connect("llm.replies", "answer_builder.replies")
    rag_pipeline_1.connect("llm.meta", "answer_builder.meta")
    rag_pipeline_1.connect("retriever", "answer_builder.documents")

    # select 5 random questions from the list of questions
    for random_questions in random.sample(questions, 5):
        question = random_questions["question"]
        answer = random_questions["answers"]["text"]
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print()
        response = rag_pipeline_1.run(
            {
                "query_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )
        print("Answer from pipeline:")
        print(response["answer_builder"]["answers"][0].data)
        print("\n")


def seven_wonders():
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    questions, documents = load_transformed_squad()
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    rag_pipeline_1 = Pipeline()
    rag_pipeline_1.add_component("retriever", InMemoryBM25Retriever(document_store, top_k=10))
    rag_pipeline_1.add_component("prompt_builder", PromptBuilder(template=template))
    rag_pipeline_1.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    rag_pipeline_1.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline_1.connect("retriever", "prompt_builder.documents")
    rag_pipeline_1.connect("prompt_builder", "llm")
    rag_pipeline_1.connect("llm.replies", "answer_builder.replies")
    rag_pipeline_1.connect("llm.meta", "answer_builder.meta")
    rag_pipeline_1.connect("retriever", "answer_builder.documents")

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    rag_pipeline_2 = Pipeline()
    rag_pipeline_2.add_component("retriever", InMemoryBM25Retriever(document_store, top_k=1))
    rag_pipeline_2.add_component("prompt_builder", PromptBuilder(template=template))
    rag_pipeline_2.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    rag_pipeline_2.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline_2.connect("retriever", "prompt_builder.documents")
    rag_pipeline_2.connect("prompt_builder", "llm")
    rag_pipeline_2.connect("llm.replies", "answer_builder.replies")
    rag_pipeline_2.connect("llm.meta", "answer_builder.meta")
    rag_pipeline_2.connect("retriever", "answer_builder.documents")

    question = "What does Rhodes Statue look like?"
    for pipeline in [rag_pipeline_1, rag_pipeline_2]:
        response = pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )
        print(response["answer_builder"]["answers"][0])

    evaluator_exact_match = AnswerExactMatchEvaluator()

    eval_pipeline = Pipeline()
    evaluator_context = RagasEvaluator(metric=RagasMetric.CONTEXT_PRECISION)
    evaluator_aspect = RagasEvaluator(
        metric=RagasMetric.ASPECT_CRITIQUE,
        metric_params={"name": "custom", "definition": "Is this answer problematic for children?", "strictness": 3},
    )
    eval_pipeline.add_component("evaluator_context", evaluator_context)
    eval_pipeline.add_component("evaluator_aspect", evaluator_aspect)
    eval_pipeline.add_component("evaluator_exact_match", evaluator_exact_match)

    questions = [
        "Where is Gardens of Babylon?",
        "Why did people build Great Pyramid of Giza?",
        # "What does Rhodes Statue look like?",
        # "Why did people visit the Temple of Artemis?",
        # "What is the importance of Colossus of Rhodes?",
        # "What happened to the Tomb of Mausolus?",
        # "How did Colossus of Rhodes collapse?",
    ]
    ground_truths = [
        "The gardens of Babylon were in the ancient city of Babylon, near present-day Hillah, Babil province, in Iraq.",
        "The pyramids of Giza were constructed to honor the pharaoh and to serve as his tomb after death.",
    ]

    for pipeline in [rag_pipeline_1, rag_pipeline_2]:
        contexts = []
        responses = []
        for question in questions:
            results = pipeline.run(
                {
                    "retriever": {"query": question},
                    "prompt_builder": {"question": question},
                    "answer_builder": {"query": question},
                }
            )

            context = [doc.content for doc in results["answer_builder"]["answers"][0].documents]
            response = results["answer_builder"]["answers"][0].data
            contexts.append(context)
            responses.append(response)

        results = eval_pipeline.run(
            {
                "evaluator_context": {"questions": questions, "contexts": contexts, "ground_truths": ground_truths},
                "evaluator_aspect": {"questions": questions, "contexts": contexts, "responses": responses},
                "evaluator_exact_match": {
                    "questions": questions,
                    "ground_truth_answers": ground_truths,
                    "predicted_answers": responses,
                },
            }
        )
        print(results)

        # Users can also run evaluator components individually outside of a pipeline
        evaluator = AnswerExactMatchEvaluator()
        exact_match_result = evaluator.run(
            questions=questions, ground_truth_answers=ground_truths, predicted_answers=responses
        )
        print(exact_match_result["result"])
