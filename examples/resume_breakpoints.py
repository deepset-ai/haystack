import json
import os
import sys

from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def indexing():
    """
    Indexing documents in a DocumentStore.
    """

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2")

    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")

    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store


def hybrid_retrieval(doc_store):
    """
    A simple pipeline for hybrid retrieval using BM25 and embeddings.
    """

    query_embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2")

    # Build a RAG pipeline with a Retriever to get relevant documents to the query and a OpenAIGenerator interacting
    # with LLMs using a custom prompt.
    prompt_template = """
    Given these documents, answer the question based on the document content only.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=doc_store), name="bm25_retriever")
    rag_pipeline.add_component(instance=query_embedder, name="query_embedder")
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=doc_store), name="embedding_retriever"
    )
    rag_pipeline.add_component(instance=DocumentJoiner(sort_by_score=False), name="doc_joiner")
    rag_pipeline.add_component(
        instance=TransformersSimilarityRanker(model="intfloat/simlm-msmarco-reranker", top_k=5), name="ranker"
    )
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

    rag_pipeline.connect("query_embedder", "embedding_retriever.query_embedding")
    rag_pipeline.connect("embedding_retriever", "doc_joiner.documents")
    rag_pipeline.connect("bm25_retriever", "doc_joiner.documents")
    rag_pipeline.connect("doc_joiner", "ranker.documents")
    rag_pipeline.connect("ranker", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("doc_joiner", "answer_builder.documents")

    return rag_pipeline


def main():
    """blah"""
    doc_store = indexing()
    pipeline = hybrid_retrieval(doc_store)

    resume_state = pipeline.load_state((sys.argv[1]))

    pipeline.run(data={}, resume_state=resume_state)


if __name__ == "__main__":
    main()
