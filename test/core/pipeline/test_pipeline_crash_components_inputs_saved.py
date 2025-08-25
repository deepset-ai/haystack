# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.core.component import component
from haystack.core.errors import PipelineError
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

# Test configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def setup_document_store():
    """Create and populate a document store with test documents."""
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)

    # Create ingestion pipeline
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store


class TestPipelineCrashStatePersistence:
    """Test pipeline crash scenarios with state persistence."""

    def test_hybrid_rag_pipeline_crash_on_embedding_retriever(self, tmp_path):
        """Test hybrid RAG pipeline crash on embedding retriever component."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Setup document store
        document_store = setup_document_store()

        # Create a mock component that returns invalid output (int instead of documents list)
        @component
        class InvalidOutputEmbeddingRetriever:
            @component.output_types(documents=list[Document])
            def run(self, query_embedding: list[float]):
                # Return an int instead of the expected documents list
                # This will cause the pipeline to crash when trying to pass it to the next component
                return 42  # Invalid output type

        # Build the hybrid RAG pipeline from scratch with the invalid retriever
        top_k = 3
        text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL, progress_bar=False)
        invalid_embedding_retriever = InvalidOutputEmbeddingRetriever()
        bm25_retriever = InMemoryBM25Retriever(document_store, top_k=top_k)
        document_joiner = DocumentJoiner(join_mode="concatenate")

        template = [
            ChatMessage.from_system(
                "You are a helpful AI assistant. Answer the following question based on the given context information "
                "only. If the context is empty or just a '\n' answer with None, example: 'None'."
            ),
            ChatMessage.from_user(
                """
                Context:
                {% for document in documents %}
                    {{ document.content }}
                {% endfor %}

                Question: {{question}}
                """
            ),
        ]

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("embedding_retriever", invalid_embedding_retriever)
        pipeline.add_component("bm25_retriever", bm25_retriever)
        pipeline.add_component("document_joiner", document_joiner)
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"])
        )
        pipeline.add_component("llm", OpenAIChatGenerator())
        pipeline.add_component("answer_builder", AnswerBuilder())

        # Connect components
        pipeline.connect("text_embedder", "embedding_retriever")
        pipeline.connect("bm25_retriever", "document_joiner")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("document_joiner.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        pipeline.connect("llm.replies", "answer_builder.replies")

        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # run pipeline and expect it to crash due to invalid output type
        with pytest.raises(PipelineError) as exc_info:
            pipeline.run(
                data=test_data,
                include_outputs_from={
                    "text_embedder",
                    "embedding_retriever",
                    "bm25_retriever",
                    "document_joiner",
                    "prompt_builder",
                    "llm",
                    "answer_builder",
                },
            )

        pipeline_outputs = exc_info.value.args[0]
        assert pipeline_outputs is not None, "Pipeline outputs should be captured in the exception"

        # verify that bm25_retriever and text_embedder ran successfully before the crash
        assert "bm25_retriever" in pipeline_outputs["serialized_data"], "BM25 retriever output not captured"
        assert "documents" in pipeline_outputs["serialized_data"]["bm25_retriever"], (
            "BM25 retriever should have produced documents"
        )
        assert "text_embedder" in pipeline_outputs["serialized_data"], "Text embedder output not captured"
        assert "embedding" in pipeline_outputs["serialized_data"]["text_embedder"], (
            "Text embedder should have produced embeddings"
        )

        # components after the crash point are not in the outputs
        assert "document_joiner" not in pipeline_outputs, "Document joiner should not have run due to crash"
        assert "prompt_builder" not in pipeline_outputs, "Prompt builder should not have run due to crash"
        assert "llm" not in pipeline_outputs, "LLM should not have run due to crash"
        assert "answer_builder" not in pipeline_outputs, "Answer builder should not have run due to crash"
