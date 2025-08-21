from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

snapshots_dir = "snapshots_complex"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"


def setup_document_store():
    """Create and populate a document store"""
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)

    # Create ingestion pipeline
    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store


def hybrid_search(document_store):
    """
    Create a hybrid RAG pipeline that combines BM25 and embedding-based retrieval.
    """
    top_k = 3
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
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


def hybrid_rag():
    """
    Example of a hybrid RAG pipeline Persisting state for components inputs only.
    """
    print(f"Using directory: {snapshots_dir}")

    print("Setting up document store...")
    document_store = setup_document_store()

    print("Creating hybrid RAG pipeline...")
    pipeline = hybrid_search(document_store)

    question = "Where does Mark live?"
    test_data = {
        "text_embedder": {"text": question},
        "bm25_retriever": {"query": question},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }

    try:
        print("Running pipeline with automatic state persistence...")
        results = pipeline.run(
            data=test_data,
            state_persistence=False,
            state_persistence_path=snapshots_dir,
            include_outputs_from={
                "answer_builder",
                "document_joiner",
                "embedding_retriever",
                "bm25_retriever",
                "llm",
                "text_embedder",
                "prompt_builder",
            },
        )
        print(f"{results['answer_builder']['answers'][0].data}")
        snapshot_files = list(Path(snapshots_dir).glob("*.json"))
        print(f"\nSnapshot files: {len(snapshot_files)}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    hybrid_rag()
