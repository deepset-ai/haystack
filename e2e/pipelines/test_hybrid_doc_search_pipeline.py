# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


def test_hybrid_doc_search_pipeline(tmp_path):
    # Create the pipeline
    document_store = InMemoryDocumentStore()
    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    hybrid_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="text_embedder"
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

    # Serialize the pipeline to YAML
    with open(tmp_path / "test_hybrid_doc_search_pipeline.yaml", "w") as f:
        hybrid_pipeline.dump(f)

    # Load the pipeline back
    with open(tmp_path / "test_hybrid_doc_search_pipeline.yaml", "r") as f:
        hybrid_pipeline = Pipeline.load(f)

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Mario and I live in the capital of Italy."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()
    embedded_documents = doc_embedder.run(documents=documents)["documents"]
    hybrid_pipeline.get_component("embedding_retriever").document_store.write_documents(embedded_documents)

    query = "Who lives in Rome?"
    result = hybrid_pipeline.run(
        {"bm25_retriever": {"query": query}, "text_embedder": {"text": query}, "ranker": {"query": query}}
    )
    assert result["ranker"]["documents"][0].content == "My name is Giorgio and I live in Rome."
    assert result["ranker"]["documents"][1].content == "My name is Mario and I live in the capital of Italy."
