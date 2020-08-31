from haystack.indexing.utils import fetch_archive_from_http
import os
from haystack.database.sql import SQLDocumentStore
from haystack.database.memory import InMemoryDocumentStore
from haystack.database.elasticsearch import Elasticsearch, ElasticsearchDocumentStore
from haystack.database.faiss import FAISSDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever, TfidfRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from time import perf_counter

from pathlib import Path


retriever_doc_stores = [("elastic", "elasticsearch"),
                        ("dpr", "faiss")]
reader_models = ["deepset/roberta-base-squad2", "deepset/minilm-uncased-squad2", "deepset/bert-base-cased-squad2", "deepset/bert-large-uncased-whole-word-masking-squad2", "deepset/xlm-roberta-large-squad2"]

reader_types = ["farm"]

data_dir = Path("../../data/nq")
filename = "nq_dev_subset_v3.json"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v3.json.zip"
doc_index = "eval_document"
label_index = "label"


def prepare_data(data_dir):
    fetch_archive_from_http(url=s3_url, output_dir=data_dir)


def get_document_store(document_store_type):
    """ TODO This method is taken from test/conftest.py but maybe should be within Haystack.
    Perhaps a class method of DocStore that just takes string for type of DocStore"""
    if document_store_type == "sql":
        if os.path.exists("haystack_test.db"):
            os.remove("haystack_test.db")
        document_store = SQLDocumentStore(url="sqlite:///haystack_test.db")
    elif document_store_type == "memory":
        document_store = InMemoryDocumentStore()
    elif document_store_type == "elasticsearch":
        # make sure we start from a fresh index
        client = Elasticsearch()
        client.indices.delete(index='haystack_test*', ignore=[404])
        document_store = ElasticsearchDocumentStore(index="eval_document")
        # ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
        #                            create_index=False, embedding_field="emb",
        #                            embedding_dim=768, excluded_meta_data=["emb"])
    elif document_store_type == "faiss":
        if os.path.exists("haystack_test_faiss.db"):
            os.remove("haystack_test_faiss.db")
        document_store = FAISSDocumentStore(sql_url="sqlite:///haystack_test_faiss.db")
    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")
    return document_store


def get_retriever(retriever_name, doc_store):
    if retriever_name == "elastic":
        return ElasticsearchRetriever(doc_store)
    if retriever_name == "tfidf":
        return TfidfRetriever(doc_store)
    if retriever_name == "dpr":
        return DensePassageRetriever(document_store=doc_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      use_gpu=True)


def get_reader(reader_name, reader_type):
    reader_class = None
    if reader_type == "farm":
        reader_class = FARMReader
    elif reader_type == "transformers":
        reader_class = TransformersReader
    return reader_class(reader_name, top_k_per_candidate=4)



def benchmark_indexing(doc_store, data_dir, filename, retriever):
    tic = perf_counter()
    index_to_doc_store(doc_store, data_dir, filename, retriever)
    toc = perf_counter()
    time = toc - tic
    return doc_store, time

def index_to_doc_store(doc_store, data_dir, filename, retriever):
    doc_store.delete_all_documents(index=doc_index)
    doc_store.delete_all_documents(index=label_index)
    doc_store.add_eval_data(data_dir / filename)
    try:
        doc_store.update_embeddings(retriever, index=doc_index)
    except AttributeError:
        pass

def main():
    retriever_results = []
    reader_results = []

    prepare_data(data_dir)
    for retriever_name, doc_store_name in retriever_doc_stores:
        doc_store = get_document_store(doc_store_name)
        retriever = get_retriever(retriever_name, doc_store)
        doc_store, indexing_time = benchmark_indexing(doc_store, data_dir, filename, retriever)
        results = retriever.eval()
        results["indexing_time"] = indexing_time
        results["retriever"] = retriever_name
        results["doc_store"] = doc_store_name
        print(results)
        retriever_results.append(results)

    doc_store = get_document_store("elasticsearch")
    index_to_doc_store(doc_store, data_dir, filename, None)
    for reader_name in reader_models:
        for reader_type in reader_types:
            reader = get_reader(reader_name, reader_type)
            results = reader.eval(document_store=doc_store,
                                  doc_index=doc_index,
                                  label_index=label_index,
                                  device="cuda")
            print(results)
            results["reader"] = reader_name
            reader_results.append(results)

    print(retriever_results)
    print(reader_results)



if __name__ == "__main__":
    main()

