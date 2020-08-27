from haystack.indexing.utils import fetch_archive_from_http
import os
from haystack.database.sql import SQLDocumentStore
from haystack.database.memory import InMemoryDocumentStore
from haystack.database.elasticsearch import Elasticsearch, ElasticsearchDocumentStore
from haystack.database.faiss import FAISSDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever, TfidfRetriever
from time import perf_counter

from pathlib import Path


retriever_doc_stores = [("elastic", "elasticsearch")]
reader_models = [""]
reader_type = ["farm", "transformers"]

data_dir = Path("../../data/nq")
filename = "nq_dev_subset_v3.json"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v3.json.zip"

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
        document_store = ElasticsearchDocumentStore(index="haystack_test")
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


def benchmark_indexing(doc_store, data_dir, filename):
    tic = perf_counter()
    doc_store.add_eval_data(data_dir / filename)
    toc = perf_counter()
    time = toc - tic
    return doc_store, time



def main():
    prepare_data(data_dir)
    for retriever_name, doc_store_name in retriever_doc_stores:
        doc_store = get_document_store(doc_store_name)
        doc_store, indexing_time = benchmark_indexing(doc_store, data_dir, filename)
        retriever = get_retriever(retriever_name, doc_store)
        results = retriever.eval()
        retriever_time = retriever.timing
        print(indexing_time)
        print(results)
        print(retriever_time)
if __name__ == "__main__":
    main()

