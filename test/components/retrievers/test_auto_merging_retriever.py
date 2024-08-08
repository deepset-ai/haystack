from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.hierarchical_doc_builder import HierarchicalDocumentBuilder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


def test_end2end():
    builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2])

    text = "one two three four five six seven eight nine ten"
    doc = Document(content=text)

    docs = builder.build_hierarchy_from_doc(doc)

    leaf_docs = [doc for doc in docs if not doc.children_ids]
    parent_docs = [doc for doc in docs if doc.children_ids]

    # store the parent documents in a document store for easy retrieval based on matching leaf/child documents
    parent_docs_store = InMemoryDocumentStore()
    parent_docs_store.write_documents(parent_docs)

    # embed and index the leaf documents
    leaf_docs_store = InMemoryDocumentStore()
    embedder = SentenceTransformersDocumentEmbedder()
    leaf_docs = embedder.run(leaf_docs)
    leaf_docs_store.write_documents(leaf_docs)

    # query the leaf documents
    query = "three four five"
    query_embedding = embedder.run([Document(content=query)])
    embedding_retrieval = InMemoryEmbeddingRetriever(document_store=leaf_docs_store)
    retrieved_leafs = embedding_retrieval.run(query_embedding=query_embedding)
