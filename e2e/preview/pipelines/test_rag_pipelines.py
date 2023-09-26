import os
import pytest

from haystack.preview import Pipeline, Document
from haystack.preview.document_stores import MemoryDocumentStore
from haystack.preview.components.writers import DocumentWriter
from haystack.preview.components.retrievers import MemoryBM25Retriever, MemoryEmbeddingRetriever
from haystack.preview.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.preview.components.generators.openai.gpt import GPTGenerator
from haystack.preview.components.builders.answer_builder import AnswerBuilder
from haystack.preview.components.builders.prompt_builder import PromptBuilder


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_bm25_rag_pipeline():
    document_store = MemoryDocumentStore()

    documents = [
        Document(text="My name is Jean and I live in Paris."),
        Document(text="My name is Mark and I live in Berlin."),
        Document(text="My name is Giorgio and I live in Rome."),
    ]

    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.text }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """

    document_store.write_documents(documents)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=MemoryBM25Retriever(document_store=document_store), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY")), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.metadata", "answer_builder.metadata")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    answers_spywords = ["Jean", "Mark", "Giorgio"]

    for question, spyword in zip(questions, answers_spywords):
        result = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        assert len(result["answer_builder"]["answers"]) == 1
        generated_answer = result["answer_builder"]["answers"][0]
        assert spyword in generated_answer.data
        assert generated_answer.query == question
        assert hasattr(generated_answer, "documents")
        assert hasattr(generated_answer, "metadata")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_embedding_retrieval_rag_pipeline():
    document_store = MemoryDocumentStore()

    documents = [
        Document(text="My name is Jean and I live in Paris."),
        Document(text="My name is Mark and I live in Berlin."),
        Document(text="My name is Giorgio and I live in Rome."),
    ]

    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.text }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/all-mpnet-base-v2"),
        name="document_embedder",
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="document_writer")
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-mpnet-base-v2"),
        name="text_embedder",
    )
    rag_pipeline.add_component(instance=MemoryEmbeddingRetriever(document_store=document_store), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY")), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.metadata", "answer_builder.metadata")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    answers_spywords = ["Jean", "Mark", "Giorgio"]

    for question, spyword in zip(questions, answers_spywords):
        result = rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        assert len(result["answer_builder"]["answers"]) == 1
        generated_answer = result["answer_builder"]["answers"][0]
        print(generated_answer)
        assert spyword in generated_answer.data
        assert generated_answer.query == question
        assert hasattr(generated_answer, "documents")
        assert hasattr(generated_answer, "metadata")
