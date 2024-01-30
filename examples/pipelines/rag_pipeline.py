import os
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner

# We are model agnostic :) Here, we use OpenAI models and load an api key from environment variables.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Create some example documents
documents = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome."),
]

# We support many different databases. Here we load a simple and lightweight in-memory document store.
document_store = InMemoryDocumentStore()

# Define some more components
doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2")
query_embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2")

ingestion_pipe = Pipeline()
ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")

ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
ingestion_pipe.run({"doc_embedder": {"documents": documents}})

# Build a RAG pipeline with a Retriever to get relevant documents to the query and a OpenAIGenerator interacting with LLMs using a custom prompt.
prompt_template = """
Given these documents, answer the question.\nDocuments:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

\nQuestion: {{question}}
\nAnswer:
"""
rag_pipeline = Pipeline()
rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
rag_pipeline.add_component(instance=query_embedder, name="query_embedder")
rag_pipeline.add_component(
    instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever"
)
rag_pipeline.add_component(instance=DocumentJoiner(sort_by_score=False), name="doc_joiner")
rag_pipeline.add_component(
    instance=TransformersSimilarityRanker(model="intfloat/simlm-msmarco-reranker", top_k=10), name="ranker"
)
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=OpenAIGenerator(api_key=OPENAI_API_KEY), name="llm")
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

# Ask a question on the data you just added.
question = "Where does Mark live?"
result = rag_pipeline.run(
    {
        "query_embedder": {"text": question},
        "bm25_retriever": {"query": question},
        "ranker": {"query": question, "top_k": 10},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }
)

# For details, like which documents were used to generate the answer, look into the GeneratedAnswer object
print(result["answer_builder"]["answers"])
