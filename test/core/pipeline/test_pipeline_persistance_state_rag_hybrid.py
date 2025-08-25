# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret


def setup_document_store(mock_doc_embedder):
    """Create and populate a document store with test documents."""
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

    # Create ingestion pipeline
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=mock_doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store


def create_hybrid_search_pipeline(document_store, mock_text_embedder, mock_llm):
    """
    Create a hybrid RAG pipeline that combines BM25 and embedding-based retrieval.
    """
    top_k = 3
    embedding_retriever = InMemoryEmbeddingRetriever(document_store, top_k=top_k)
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

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", mock_text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component(
        "prompt_builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"])
    )
    hybrid_retrieval.add_component("llm", mock_llm)
    hybrid_retrieval.add_component("answer_builder", AnswerBuilder())
    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner.documents", "prompt_builder.documents")
    hybrid_retrieval.connect("prompt_builder", "llm")
    hybrid_retrieval.connect("llm.replies", "answer_builder.replies")

    return hybrid_retrieval


def load_snapshot_files(snapshots_dir: Path) -> dict[str, Any]:
    """
    Load and parse all JSON snapshot files in the specified directory.

    :param snapshots_dir: Directory containing snapshot files
    :return: Dictionary mapping component names to their snapshot data
    """
    snapshots = {}
    for snapshot_file in snapshots_dir.glob("*.json"):
        try:
            snapshot = load_pipeline_snapshot(snapshot_file)
            component_name = snapshot.break_point.component_name
            if component_name not in snapshots:
                snapshots[component_name] = []
            snapshots[component_name].append(
                {
                    "file": snapshot_file.name,
                    "visit_count": snapshot.break_point.visit_count,
                    "timestamp": snapshot.timestamp,
                    "snapshot": snapshot,
                }
            )
        except Exception as e:
            print(f"Error loading snapshot {snapshot_file.name}: {e}")

    return snapshots


class TestHybridRAGStatePersistence:
    @pytest.fixture
    def mock_sentence_transformers_doc_embedder(self):
        with patch(
            "haystack.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"
        ) as mock_doc_embedder:
            mock_model = MagicMock()
            mock_doc_embedder.return_value = mock_model

            def mock_encode(
                documents, batch_size=None, show_progress_bar=None, normalize_embeddings=None, precision=None, **kwargs
            ):  # noqa E501
                import numpy as np

                return [np.ones(384).tolist() for _ in documents]

            mock_model.encode = mock_encode
            embedder = SentenceTransformersDocumentEmbedder(model="mock-model", progress_bar=False)

            def mock_run(documents: list[Document]):
                if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
                    raise TypeError(
                        "SentenceTransformersDocumentEmbedder expects a list of Documents as input."
                        "In case you want to embed a string, please use the SentenceTransformersTextEmbedder."
                    )

                import numpy as np

                embedding = np.ones(384).tolist()

                for doc in documents:
                    doc.embedding = embedding

                return {"documents": documents}

            embedder.run = mock_run
            embedder.warm_up()
            return embedder

    @pytest.fixture
    def mock_sentence_transformers_text_embedder(self):
        with patch(
            "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
        ) as mock_text_embedder:
            mock_model = MagicMock()
            mock_text_embedder.return_value = mock_model

            def mock_encode(
                texts, batch_size=None, show_progress_bar=None, normalize_embeddings=None, precision=None, **kwargs
            ):  # noqa E501
                import numpy as np

                return [np.ones(384).tolist() for _ in texts]

            mock_model.encode = mock_encode
            embedder = SentenceTransformersTextEmbedder(model="mock-model", progress_bar=False)

            def mock_run(text):
                if not isinstance(text, str):
                    raise TypeError(
                        "SentenceTransformersTextEmbedder expects a string as input."
                        "In case you want to embed a list of Documents, please use the "
                        "SentenceTransformersDocumentEmbedder."
                    )

                import numpy as np

                embedding = np.ones(384).tolist()
                return {"embedding": embedding}

            embedder.run = mock_run
            embedder.warm_up()
            return embedder

    @pytest.fixture
    def mock_openai_completion(self):
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            mock_completion = MagicMock()
            mock_completion.model = "gpt-4o-mini"
            mock_completion.choices = [
                MagicMock(finish_reason="stop", index=0, message=MagicMock(content="Mark lives in Berlin."))
            ]
            mock_completion.usage = {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97}

            mock_chat_completion_create.return_value = mock_completion
            yield mock_chat_completion_create

    @pytest.mark.integration
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})
    def test_hybrid_rag_pipeline_with_state_persistence(
        self,
        tmp_path,
        mock_sentence_transformers_doc_embedder,
        mock_sentence_transformers_text_embedder,
        mock_openai_completion,
    ):
        """Test that hybrid RAG pipeline creates snapshots with state persistence enabled."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Setup document store with mocked embedder
        document_store = setup_document_store(mock_sentence_transformers_doc_embedder)

        # Create the pipeline with mocked components
        mock_llm = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"))
        pipeline = create_hybrid_search_pipeline(document_store, mock_sentence_transformers_text_embedder, mock_llm)

        # Test data
        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # Run pipeline with automatic state persistence
        results = pipeline.run(data=test_data, state_persistence=True, state_persistence_path=str(snapshots_dir))

        # Verify pipeline completed successfully
        assert "answer_builder" in results, "Pipeline should complete successfully"
        assert "answers" in results["answer_builder"], "Answer builder should produce answers"
        assert len(results["answer_builder"]["answers"]) > 0, "Should have at least one answer"

        # Check if snapshot files were created
        snapshot_files = list(snapshots_dir.glob("*.json"))
        assert len(snapshot_files) > 0, "Snapshot files should be created"

        for f_name in snapshot_files:
            print(f"Snapshot file created: {f_name.name}")

        # Load and verify snapshot data
        snapshots = load_snapshot_files(snapshots_dir)

        # Verify that snapshots exist for key components
        expected_components = [
            "text_embedder",
            "embedding_retriever",
            "bm25_retriever",
            "document_joiner",
            "prompt_builder",
            "llm",
        ]
        for component_name in expected_components:
            assert component_name in snapshots, f"Should have snapshots for {component_name}"
            assert len(snapshots[component_name]) > 0, f"Should have at least one snapshot for {component_name}"

    @pytest.mark.integration
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})
    def test_hybrid_rag_pipeline_resume_from_snapshot(
        self,
        tmp_path,
        mock_sentence_transformers_doc_embedder,
        mock_sentence_transformers_text_embedder,
        mock_openai_completion,
    ):
        """Test that pipeline can be resumed from a snapshot."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Setup document store with mocked embedder
        document_store = setup_document_store(mock_sentence_transformers_doc_embedder)

        # Create the pipeline with mocked components
        mock_llm = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"))
        pipeline = create_hybrid_search_pipeline(document_store, mock_sentence_transformers_text_embedder, mock_llm)

        # Test data
        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # Run pipeline with state persistence
        _ = pipeline.run(data=test_data, state_persistence=True, state_persistence_path=str(snapshots_dir))

        # Load snapshots
        snapshots = load_snapshot_files(snapshots_dir)

        # Test resuming from each component's latest snapshot
        for component_name, component_snapshots in snapshots.items():
            if component_snapshots:
                # Get the latest snapshot for this component
                latest_snapshot = max(component_snapshots, key=lambda x: x["timestamp"])
                snapshot = latest_snapshot["snapshot"]

                # Resume from snapshot
                resumed_results = pipeline.run(
                    data={},  # Empty data since we're resuming
                    pipeline_snapshot=snapshot,
                )

                # Verify that resuming produces results
                assert resumed_results is not None, f"Resuming from {component_name} should produce results"

                # If we resumed from a point that should produce final answers, verify they exist
                if "answer_builder" in resumed_results:
                    assert "answers" in resumed_results["answer_builder"], "Resumed pipeline should produce answers"
                    assert len(resumed_results["answer_builder"]["answers"]) > 0, "Should have at least one answer"
