"""
This script performs the same query benchmarking as `retriever.py` but with less of the loops that iterate
over all the parameters so that it is easier to inspect what is happening
"""


from haystack.document_stores import MilvusDocumentStore, FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from retriever import prepare_data
import datetime
from pprint import pprint
from milvus import IndexType
from utils import get_document_store


def benchmark_querying(index_type, n_docs=100_000, similarity="dot_product"):
    doc_index = "document"
    label_index = "label"

    docs, labels = prepare_data(
        data_dir="data/",
        filename_gold="nq2squad-dev.json",
        filename_negative="psgs_w100_minus_gold_100k.tsv",
        remote_url="https://ext-haystack-retriever-eval.s3-eu-west-1.amazonaws.com/",
        embeddings_filenames=["wikipedia_passages_100k.pkl"],
        embeddings_dir="embeddings/",
        n_docs=n_docs,
        add_precomputed=True,
    )

    doc_store = get_document_store(document_store_type=index_type, similarity=similarity)

    # if index_type == "milvus_flat":
    #     doc_store = MilvusDocumentStore(index=doc_index, similarity=similarity)
    # elif index_type == "milvus_hnsw":
    #     index_param = {"M": 64, "efConstruction": 80}
    #     search_param = {"ef": 20}
    #     doc_store = MilvusDocumentStore(
    #         index=doc_index,
    #         index_type=IndexType.HNSW,
    #         index_param=index_param,
    #         search_param=search_param,
    #         similarity=similarity
    #     )

    doc_store.write_documents(documents=docs, index=doc_index)
    doc_store.write_labels(labels=labels, index=label_index)

    retriever = DensePassageRetriever(
        document_store=doc_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=True,
        use_fast_tokenizers=True,
    )

    raw_results = retriever.eval(label_index=label_index, doc_index=doc_index)
    results = {
        "n_queries": raw_results["n_questions"],
        "retrieve_time": raw_results["retrieve_time"],
        "queries_per_second": raw_results["n_questions"] / raw_results["retrieve_time"],
        "seconds_per_query": raw_results["retrieve_time"] / raw_results["n_questions"],
        "recall": raw_results["recall"] * 100,
        "map": raw_results["map"] * 100,
        "top_k": raw_results["top_k"],
        "date_time": datetime.datetime.now(),
        "error": None,
    }

    pprint(results)

    doc_store.delete_all_documents(index=doc_index)
    doc_store.delete_all_documents(index=label_index)


if __name__ == "__main__":
    similarity = "l2"
    n_docs = 1000

    benchmark_querying(index_type="milvus_flat", similarity=similarity, n_docs=n_docs)
    benchmark_querying(index_type="milvus_hnsw", similarity=similarity, n_docs=n_docs)
    benchmark_querying(index_type="faiss_flat", similarity=similarity, n_docs=n_docs)
    benchmark_querying(index_type="faiss_hnsw", similarity=similarity, n_docs=n_docs)
