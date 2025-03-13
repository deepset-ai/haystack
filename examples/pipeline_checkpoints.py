import os
import threading
import time

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
    """
    Test
    """

    doc_store = indexing()
    pipeline = hybrid_retrieval(doc_store)

    question = "Where does Mark live?"
    data = {
        "query_embedder": {"text": question},
        "bm25_retriever": {"query": question},
        "ranker": {"query": question, "top_k": 10},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }

    print("Running pipeline without breakpoints")
    result = pipeline.run(data)
    print(result["answer_builder"]["answers"])

    # Let's add a breakpoint in the pipeline
    print("Adding a breakpoint to 'doc_joiner'")
    pipeline.add_breakpoint("doc_joiner")
    print("Breakpoints:", pipeline.list_breakpoints())

    # function to run the pipeline in a separate thread
    def run_pipeline_thread():
        global results
        try:
            results = pipeline.run(data)
            print("Pipeline execution completed!")
        except Exception as e:
            print(f"Pipeline execution failed: {e}")

    # launching/running the pipeline in a separate thread
    pipeline_thread = threading.Thread(target=run_pipeline_thread)
    pipeline_thread.daemon = True
    pipeline_thread.start()

    # wait for the pipeline to pause at the breakpoint
    max_wait_time = 30
    start_time = time.time()

    # controlling thread now waits for the pipeline to pause at the breakpoint
    while not pipeline.is_paused():
        time.sleep(0.1)  # Small delay to prevent CPU spinning

        # stop if we waited too long or the pipeline thread ended unexpectedly
        if time.time() - start_time > max_wait_time:
            print("Timeout waiting for breakpoint")
            break

        if not pipeline_thread.is_alive():
            print("Pipeline thread ended unexpectedly")
            break

    if pipeline.is_paused():
        print(f"Pipeline paused at: {pipeline.get_pause_state()}")

        state = pipeline.get_current_state()
        print(f"Current state: {state}")

        _ = input("Press Enter to continue execution...")

        pipeline.resume()

        # need to join/wait for the pipeline to complete
        pipeline_thread.join()

        # Now we can access the final results
        if "results" in globals():
            print("Final results:", results["answer_builder"]["answers"])
    else:
        print("Pipeline did not pause at the breakpoint")


if __name__ == "__main__":
    main()
