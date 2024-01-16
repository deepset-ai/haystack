import os
import json
import pytest

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_bm25_rag_pipeline(tmp_path):
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
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=OpenAIGenerator(api_key=os.environ.get("OPENAI_API_KEY")), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Draw the pipeline
    rag_pipeline.draw(tmp_path / "test_bm25_rag_pipeline.png")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_bm25_rag_pipeline.json", "w") as f:
        json.dump(rag_pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_bm25_rag_pipeline.json", "r") as f:
        rag_pipeline = Pipeline.from_dict(json.load(f))

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    rag_pipeline.get_component("retriever").document_store.write_documents(documents)

    # Query and assert
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
        assert hasattr(generated_answer, "meta")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
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
    rag_pipeline.add_component(instance=OpenAIGenerator(api_key=os.environ.get("OPENAI_API_KEY")), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Draw the pipeline
    rag_pipeline.draw(tmp_path / "test_embedding_rag_pipeline.png")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_embedding_rag_pipeline.json", "w") as f:
        json.dump(rag_pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_embedding_rag_pipeline.json", "r") as f:
        rag_pipeline = Pipeline.from_dict(json.load(f))

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
        assert spyword in generated_answer.data
        assert generated_answer.query == question
        assert hasattr(generated_answer, "documents")
        assert hasattr(generated_answer, "meta")
