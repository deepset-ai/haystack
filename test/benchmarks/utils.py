import os
from haystack import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.elasticsearch import Elasticsearch, ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever, TfidfRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from time import perf_counter
import pandas as pd
import json

from pathlib import Path


reader_models = ["deepset/roberta-base-squad2", "deepset/minilm-uncased-squad2", "deepset/bert-base-cased-squad2", "deepset/bert-large-uncased-whole-word-masking-squad2", "deepset/xlm-roberta-large-squad2"]
reader_types = ["farm"]
data_dir_reader = Path("../../data/squad20")
filename_reader = "dev-v2.0.json"

doc_index = "eval_document"
label_index = "label"

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


def get_reader(reader_name, reader_type, max_seq_len=384):
    reader_class = None
    if reader_type == "farm":
        reader_class = FARMReader
    elif reader_type == "transformers":
        reader_class = TransformersReader
    return reader_class(reader_name, top_k_per_candidate=4, max_seq_len=max_seq_len)

def perform_retriever_eval():
    retriever_results = []
    for retriever_name, doc_store_name in retriever_doc_stores:
        # try:
        doc_store = get_document_store(doc_store_name)
        retriever = get_retriever(retriever_name, doc_store)
        doc_store, indexing_time = benchmark_indexing(doc_store, data_dir_retriever, filename_retriever, retriever)
        results = retriever.eval()
        results["indexing_time"] = indexing_time
        results["retriever"] = retriever_name
        results["doc_store"] = doc_store_name
        print(results)
        retriever_results.append(results)
        # except Exception as e:
        #     retriever_results.append(str(e))

    retriever_df = pd.DataFrame.from_records(retriever_results)
    retriever_df.to_csv("retriever_results.csv")

