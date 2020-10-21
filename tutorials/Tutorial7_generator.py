import os
import pathlib

import torch
from datasets import load_from_disk
from transformers import RagRetriever

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


def fetch_from_transformer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    passages_path = os.path.join(pathlib.Path().absolute(),
                                 "sample_transformers_data/my_knowledge_dataset")
    dataset = load_from_disk(passages_path)
    index_path = os.path.join(pathlib.Path().absolute(),
                              "sample_transformers_data/my_knowledge_dataset_hnsw_index.faiss")
    dataset.load_faiss_index("embeddings", index_path)

    # Adding transformers part to verify
    # Question tokenization
    input_dict = haystack_generator.tokenizer.prepare_seq2seq_batch(
        src_texts=[question],
        return_tensors="pt"
    )

    rag_retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq",
        index_name="custom",
        indexed_dataset=dataset
    )

    haystack_generator.model.set_retriever(rag_retriever)
    rag_model = haystack_generator.model

    generated = rag_model.generate(input_dict["input_ids"])
    generated_string = haystack_generator.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print("By Transformers=", generated_string)


fetch_from_transformer()
