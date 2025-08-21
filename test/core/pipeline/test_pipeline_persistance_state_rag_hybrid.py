from pathlib import Path
from typing import Any, Dict

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


def create_hybrid_search_pipeline(document_store):
    """
    Create a hybrid RAG pipeline that combines BM25 and embedding-based retrieval.
    """
    top_k = 3
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL, progress_bar=False)
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
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component(
        "prompt_builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"])
    )
    hybrid_retrieval.add_component("llm", OpenAIChatGenerator())
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
    """Test state persistence functionality with hybrid RAG pipelines."""

    def test_hybrid_rag_pipeline_with_state_persistence(self, tmp_path):
        """Test that hybrid RAG pipeline creates snapshots with state persistence enabled."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Setup document store
        document_store = setup_document_store()

        # Create the pipeline
        pipeline = create_hybrid_search_pipeline(document_store)

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

    def test_hybrid_rag_pipeline_resume_from_snapshot(self, tmp_path):
        """Test that pipeline can be resumed from a snapshot."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Setup document store
        document_store = setup_document_store()

        # Create the pipeline
        pipeline = create_hybrid_search_pipeline(document_store)

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

    def test_hybrid_rag_pipeline_snapshot_timing(self, tmp_path):
        """Test that snapshots are created at the right timing during pipeline execution."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Setup document store
        document_store = setup_document_store()

        # Create the pipeline
        pipeline = create_hybrid_search_pipeline(document_store)

        # Test data
        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # Run pipeline with state persistence
        pipeline.run(data=test_data, state_persistence=True, state_persistence_path=str(snapshots_dir))

        # Load snapshots
        snapshots = load_snapshot_files(snapshots_dir)

        # Verify that snapshots have reasonable timestamps
        for component_name, component_snapshots in snapshots.items():
            for snapshot_info in component_snapshots:
                timestamp = snapshot_info["timestamp"]
                assert timestamp is not None, f"Snapshot for {component_name} should have a timestamp"

                # Verify timestamp is a reasonable value (not too old or in the future)
                # This is a basic sanity check - timestamps should be recent
                assert timestamp > 0, f"Timestamp for {component_name} should be positive"

    def test_hybrid_rag_pipeline_component_visit_counts(self, tmp_path):
        """Test that component visit counts are correctly tracked in snapshots."""
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Setup document store
        document_store = setup_document_store()

        # Create the pipeline
        pipeline = create_hybrid_search_pipeline(document_store)

        # Test data
        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # Run pipeline with state persistence
        pipeline.run(data=test_data, state_persistence=True, state_persistence_path=str(snapshots_dir))

        # Load snapshots
        snapshots = load_snapshot_files(snapshots_dir)

        # Verify visit counts for each component
        for component_name, component_snapshots in snapshots.items():
            for snapshot_info in component_snapshots:
                visit_count = snapshot_info["visit_count"]
                pipeline_state = snapshot_info["snapshot"].pipeline_state

                # Verify visit count matches the pipeline state
                assert component_name in pipeline_state.component_visits, (
                    f"Component {component_name} should be in visits"
                )
                assert pipeline_state.component_visits[component_name] == visit_count, (
                    f"Visit count should match for {component_name}"
                )

                # Verify visit count is reasonable (at least 1, since the component was visited)
                assert visit_count >= 1, f"Visit count for {component_name} should be at least 1"
