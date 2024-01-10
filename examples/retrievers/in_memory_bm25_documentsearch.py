from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.pipeline import Pipeline

# Create components and a query pipeline
document_store = InMemoryDocumentStore()
retriever = InMemoryBM25Retriever(document_store=document_store)

pipeline = Pipeline()
pipeline.add_component(instance=retriever, name="retriever")

# Add Documents
documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(
        content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."
    ),
    Document(
        content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves."
    ),
]
document_store.write_documents(documents)

# Run the pipeline
result = pipeline.run(data={"retriever": {"query": "How many languages are there?"}})

print(result["retriever"]["documents"][0])
