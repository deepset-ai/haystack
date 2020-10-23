from haystack import Document
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.generator.transformers import RAGenerator
from haystack.retriever.dense import DensePassageRetriever

# Add documents from which you want generate answers
documents = [
    Document(
        text="""The capital of Germany is the city state of Berlin."""
    ),
    Document(
        text="""Berlin is the capital and largest city of Germany by both area and population.""",
    )
]

# Initialize FAISS document store to documents and corresponding index for embeddings
document_store = FAISSDocumentStore()

# Initialize DPR Retriever to encode documents, encode question and query documents
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False,
    embed_title=True,
    remove_sep_tok_from_untitled_passages=True
)

# Initialize RAG Generator
generator = RAGenerator(
    model_name_or_path="facebook/rag-token-nq",
    retriever=retriever,
    use_gpu=False,
    top_k_answers=1,
    max_length=200,
    min_length=2,
    embed_title=True,
    num_beams=2,
)

# Delete existing documents in documents store
document_store.delete_all_documents()
# Write documents to document store
document_store.write_documents(documents)
# Add documents embeddings to index
document_store.update_embeddings(
    retriever=retriever
)

# Now ask your question and retrieve related documents from retriever
question = "What is capital of the Germany?"
retriever_results = retriever.retrieve(
    query=question
)

# Now generate answer from question and retrieved documents
predicted_result = generator.predict(
    question=question,
    documents=retriever_results,
    top_k=1
)

# Print you answer
answers = predicted_result["answers"]
for idx, answer in enumerate(answers):
    print(f'Generated answer#{idx + 1} is{answer["answer"]}')
