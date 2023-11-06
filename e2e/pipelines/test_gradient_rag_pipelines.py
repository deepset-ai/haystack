import os
import json
import pytest

from haystack.preview import Pipeline, Document
from haystack.preview.components.embedders.gradient_document_embedder import GradientDocumentEmbedder
from haystack.preview.components.embedders.gradient_text_embedder import GradientTextEmbedder
from haystack.preview.document_stores import InMemoryDocumentStore
from haystack.preview.components.writers import DocumentWriter
from haystack.preview.components.retrievers import InMemoryEmbeddingRetriever
from haystack.preview.components.generators.gradient.base import GradientGenerator
from haystack.preview.components.builders.answer_builder import AnswerBuilder
from haystack.preview.components.builders.prompt_builder import PromptBuilder


@pytest.mark.skipif(
    not os.environ.get("GRADIENT_ACCESS_TOKEN", None) or not os.environ.get("GRADIENT_WORKSPACE_ID", None),
    reason="Export env variables called GRADIENT_ACCESS_TOKEN and GRADIENT_WORKSPACE_ID containing the Gradient configuration settings to run this test.",
)
def test_gradient_embedding_retrieval_rag_pipeline(tmp_path):
    # Create the RAG pipeline
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """

    gradient_access_token = os.environ.get("GRADIENT_ACCESS_TOKEN")
    rag_pipeline = Pipeline()
    embedder = GradientTextEmbedder(access_token=gradient_access_token)
    rag_pipeline.add_component(instance=embedder, name="text_embedder")
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever"
    )
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=GradientGenerator(access_token=gradient_access_token, base_model_slug="llama2-7b-chat"), name="llm"
    )
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Draw the pipeline
    rag_pipeline.draw(tmp_path / "test_gradient_embedding_rag_pipeline.png")

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
    document_store = rag_pipeline.get_component("retriever").document_store
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=GradientDocumentEmbedder(), name="document_embedder")
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
        assert hasattr(generated_answer, "metadata")
