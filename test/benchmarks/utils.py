import os
from haystack.document_stores import SQLDocumentStore
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore, OpenSearchDocumentStore
from haystack.document_stores.elasticsearch import Elasticsearch
from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import MilvusDocumentStore
from haystack.nodes import BM25Retriever, TfidfRetriever
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.nodes import TransformersReader
from haystack.utils import launch_milvus, launch_es, launch_opensearch
from haystack.modeling.data_handler.processor import http_get

import logging
import subprocess
import time
import json
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)


reader_models = [
    "deepset/roberta-base-squad2",
    "deepset/minilm-uncased-squad2",
    "deepset/bert-base-cased-squad2",
    "deepset/bert-large-uncased-whole-word-masking-squad2",
    "deepset/xlm-roberta-large-squad2",
]
reader_types = ["farm"]

doc_index = "eval_document"
label_index = "label"


def get_document_store(document_store_type, similarity="dot_product", index="document"):
    """TODO This method is taken from test/conftest.py but maybe should be within Haystack.
    Perhaps a class method of DocStore that just takes string for type of DocStore"""
    if document_store_type == "sql":
        if os.path.exists("haystack_test.db"):
            os.remove("haystack_test.db")
        document_store = SQLDocumentStore(url="sqlite:///haystack_test.db")
        assert document_store.get_document_count() == 0
    elif document_store_type == "memory":
        document_store = InMemoryDocumentStore()
    elif document_store_type == "elasticsearch":
        launch_es()
        time.sleep(5)
        # make sure we start from a fresh index
        client = Elasticsearch()
        client.indices.delete(index="haystack_test*", ignore=[404])
        document_store = ElasticsearchDocumentStore(index="eval_document", similarity=similarity, timeout=3000)
    elif document_store_type in ("milvus_flat", "milvus_hnsw"):
        launch_milvus()
        if document_store_type == "milvus_flat":
            index_type = "FLAT"
            index_param = None
            search_param = None
        elif document_store_type == "milvus_hnsw":
            index_type = "HNSW"
            index_param = {"M": 64, "efConstruction": 80}
            search_param = {"ef": 20}
        document_store = MilvusDocumentStore(
            similarity=similarity,
            index_type=index_type,
            index_param=index_param,
            search_param=search_param,
            index=index,
        )
        assert document_store.get_document_count(index="eval_document") == 0
    elif document_store_type in ("faiss_flat", "faiss_hnsw"):
        if document_store_type == "faiss_flat":
            index_type = "Flat"
        elif document_store_type == "faiss_hnsw":
            index_type = "HNSW"
        status = subprocess.run(["docker rm -f haystack-postgres"], shell=True)
        time.sleep(1)
        status = subprocess.run(
            ["docker run --name haystack-postgres -p 5432:5432 -e POSTGRES_PASSWORD=password -d postgres"], shell=True
        )
        time.sleep(6)
        status = subprocess.run(
            ['docker exec haystack-postgres psql -U postgres -c "CREATE DATABASE haystack;"'], shell=True
        )
        time.sleep(1)
        document_store = FAISSDocumentStore(
            sql_url="postgresql://postgres:password@localhost:5432/haystack",
            faiss_index_factory_str=index_type,
            similarity=similarity,
            index=index,
        )
        assert document_store.get_document_count() == 0
    elif document_store_type in ("opensearch_flat", "opensearch_hnsw"):
        launch_opensearch(local_port=9201)
        if document_store_type == "opensearch_flat":
            index_type = "flat"
        elif document_store_type == "opensearch_hnsw":
            index_type = "hnsw"
        document_store = OpenSearchDocumentStore(index_type=index_type, port=9201, timeout=3000)
    else:
        raise Exception(f"No document store fixture for '{document_store_type}'")
    return document_store


def get_retriever(retriever_name, doc_store, devices):
    if retriever_name == "elastic":
        return BM25Retriever(doc_store)
    if retriever_name == "tfidf":
        return TfidfRetriever(doc_store)
    if retriever_name == "dpr":
        return DensePassageRetriever(
            document_store=doc_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=True,
            use_fast_tokenizers=False,
            devices=devices,
        )
    if retriever_name == "sentence_transformers":
        return EmbeddingRetriever(
            document_store=doc_store,
            embedding_model="nq-distilbert-base-v1",
            use_gpu=True,
            model_format="sentence_transformers",
        )


def get_reader(reader_name, reader_type, max_seq_len=384):
    reader_class = None
    if reader_type == "farm":
        reader_class = FARMReader
    elif reader_type == "transformers":
        reader_class = TransformersReader
    return reader_class(reader_name, top_k_per_candidate=4, max_seq_len=max_seq_len)


def index_to_doc_store(doc_store, docs, retriever, labels=None):
    doc_store.write_documents(docs, doc_index)
    if labels:
        doc_store.write_labels(labels, index=label_index)
    # these lines are not run if the docs.embedding field is already populated with precomputed embeddings
    # See the prepare_data() fn in the retriever benchmark script
    if callable(getattr(retriever, "embed_documents", None)) and docs[0].embedding is None:
        doc_store.update_embeddings(retriever, index=doc_index, batch_size=200)


def load_config(config_filename, ci):
    conf = json.load(open(config_filename))
    if ci:
        params = conf["params"]["ci"]
    else:
        params = conf["params"]["full"]
    filenames = conf["filenames"]
    max_docs = max(params["n_docs_options"])
    n_docs_keys = sorted([int(x) for x in list(filenames["embeddings_filenames"])])
    for k in n_docs_keys:
        if max_docs <= k:
            filenames["embeddings_filenames"] = [filenames["embeddings_filenames"][str(k)]]
            filenames["filename_negative"] = filenames["filenames_negative"][str(k)]
            break
    return params, filenames


def download_from_url(url: str, filepath: Union[str, Path]):
    """
    Download from a url to a local file. Skip already existing files.

    :param url: Url
    :param filepath: local path where the url content shall be stored
    :return: local path of the downloaded file
    """

    logger.info("Downloading %s", url)
    # Create local folder
    folder, filename = os.path.split(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download file if not present locally
    if os.path.exists(filepath):
        logger.info("Skipping %s (exists locally)", url)
    else:
        logger.info("Downloading %s to %s", filepath)
        with open(filepath, "wb") as file:
            http_get(url=url, temp_file=file)
    return filepath
