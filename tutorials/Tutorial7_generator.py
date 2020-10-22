from haystack import Document
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.generator.transformers import RAGenerator
from haystack.retriever.dense import DensePassageRetriever

# haystack part

documents = [
    Document(
        text="""Berlin is Germany capital"""
    ),
    Document(
        text="""Berlin is the capital and largest city of Germany by both area and population.""",
    )
]

document_store = FAISSDocumentStore(faiss_index_factory_str="HNSW")
document_store.delete_all_documents()
document_store.write_documents(documents)

retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  use_gpu=False, embed_title=True,
                                  remove_sep_tok_from_untitled_passages=True)

document_store.update_embeddings(retriever=retriever)

docs_with_emb = document_store.get_all_documents()

question = "What is capital of the Germany?"
retriever_results = retriever.retrieve(query=question, top_k=2)

haystack_generator = RAGenerator(retriever=retriever)
predicted_result = haystack_generator.predict(question=question, documents=retriever_results, top_k=1)

print("By Haystack=", predicted_result["answers"][0]["answer"])
