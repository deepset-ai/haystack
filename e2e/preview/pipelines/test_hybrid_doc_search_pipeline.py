import json

from haystack.preview import Pipeline, Document
from haystack.preview.components.embedders import SentenceTransformersTextEmbedder
from haystack.preview.components.rankers import TransformersSimilarityRanker
from haystack.preview.components.routers.document_joiner import DocumentJoiner
from haystack.preview.document_stores import InMemoryDocumentStore
from haystack.preview.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever


def test_hybrid_doc_search_pipeline(tmp_path):
    # Create the pipeline
    document_store = InMemoryDocumentStore()
    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    hybrid_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="text_embedder",
    )
    hybrid_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever"
    )
    hybrid_pipeline.add_component(instance=DocumentJoiner(), name="joiner")
    hybrid_pipeline.add_component(instance=TransformersSimilarityRanker(top_k=20), name="ranker")

    hybrid_pipeline.connect("bm25_retriever", "joiner")
    hybrid_pipeline.connect("text_embedder", "embedding_retriever")
    hybrid_pipeline.connect("embedding_retriever", "joiner")
    hybrid_pipeline.connect("joiner", "ranker")

    # Draw the pipeline
    hybrid_pipeline.draw(tmp_path / "test_hybrid_doc_search_pipeline.png")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_hybrid_doc_search_pipeline.json", "w") as f:
        print(json.dumps(hybrid_pipeline.to_dict(), indent=4))
        json.dump(hybrid_pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_hybrid_doc_search_pipeline.json", "r") as f:
        hybrid_pipeline = Pipeline.from_dict(json.load(f))

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Mario and I live in the capital of Italy."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    hybrid_pipeline.get_component("bm25_retriever").document_store.write_documents(documents)

    query = "Who lives in Rome?"
    result = hybrid_pipeline.run(
        {"bm25_retriever": {"query": query}, "text_embedder": {"text": query}, "ranker": {"query": query}}
    )
    assert result["ranker"]["documents"][0].content == "My name is Giorgio and I live in Rome."
    assert result["ranker"]["documents"][1].content == "My name is Mario and I live in the capital of Italy."
