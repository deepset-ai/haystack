import pandas as pd
from pathlib import Path
from time import perf_counter
from utils import get_document_store, get_retriever, index_to_doc_store, load_config
from haystack.preprocessor.utils import eval_data_from_file
from haystack import Document
import pickle
import time
from tqdm import tqdm
import logging
import datetime
import random
import traceback
import os
import requests
from farm.file_utils import download_from_s3
import json



logger = logging.getLogger(__name__)
logging.getLogger("haystack.retriever.base").setLevel(logging.WARN)
logging.getLogger("elasticsearch").setLevel(logging.WARN)

doc_index = "eval_document"
label_index = "label"

seed = 42
random.seed(42)

def benchmark_indexing(n_docs_options, retriever_doc_stores, data_dir, filename_gold, filename_negative, data_s3_url, embeddings_filenames, embeddings_dir, **kwargs):

    retriever_results = []
    for n_docs in n_docs_options:
        for retriever_name, doc_store_name in retriever_doc_stores:
            logger.info(f"##### Start indexing run: {retriever_name}, {doc_store_name}, {n_docs} docs ##### ")
            try:
                doc_store = get_document_store(doc_store_name)
                retriever = get_retriever(retriever_name, doc_store)
                docs, _ = prepare_data(data_dir=data_dir,
                                       filename_gold=filename_gold,
                                       filename_negative=filename_negative,
                                       data_s3_url=data_s3_url,
                                       embeddings_filenames=embeddings_filenames,
                                       embeddings_dir=embeddings_dir,
                                       n_docs=n_docs)

                tic = perf_counter()
                index_to_doc_store(doc_store, docs, retriever)
                toc = perf_counter()
                indexing_time = toc - tic

                print(indexing_time)

                retriever_results.append({
                    "retriever": retriever_name,
                    "doc_store": doc_store_name,
                    "n_docs": n_docs,
                    "indexing_time": indexing_time,
                    "docs_per_second": n_docs / indexing_time,
                    "date_time": datetime.datetime.now(),
                    "error": None})
                retriever_df = pd.DataFrame.from_records(retriever_results)
                retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
                retriever_df.to_csv("retriever_index_results.csv")
                doc_store.delete_all_documents(index=doc_index)
                doc_store.delete_all_documents(index=label_index)
                time.sleep(10)
                del doc_store
                del retriever

            except Exception as e:
                tb = traceback.format_exc()
                retriever_results.append({
                    "retriever": retriever_name,
                    "doc_store": doc_store_name,
                    "n_docs": n_docs,
                    "indexing_time": 0,
                    "docs_per_second": 0,
                    "date_time": datetime.datetime.now(),
                    "error": str(tb)})
                doc_store.delete_all_documents(index=doc_index)
                doc_store.delete_all_documents(index=label_index)
                time.sleep(10)
                del doc_store
                del retriever

def benchmark_querying(n_docs_options,
                       retriever_doc_stores,
                       data_dir,
                       data_s3_url,
                       filename_gold,
                       filename_negative,
                       n_queries,
                       embeddings_filenames,
                       embeddings_dir,
                       **kwargs):
    """ Benchmark the time it takes to perform querying. Doc embeddings are loaded from file."""
    retriever_results = []

    for n_docs in n_docs_options:
        for retriever_name, doc_store_name in retriever_doc_stores:
            try:
                logger.info(f"##### Start querying run: {retriever_name}, {doc_store_name}, {n_docs} docs ##### ")
                doc_store = get_document_store(doc_store_name)
                retriever = get_retriever(retriever_name, doc_store)
                add_precomputed = retriever_name in ["dpr"]
                # For DPR, precomputed embeddings are loaded from file
                docs, labels = prepare_data(data_dir=data_dir,
                                            filename_gold=filename_gold,
                                            filename_negative=filename_negative,
                                            data_s3_url=data_s3_url,
                                            embeddings_filenames=embeddings_filenames,
                                            embeddings_dir=embeddings_dir,
                                            n_docs=n_docs,
                                            n_queries=n_queries,
                                            add_precomputed=add_precomputed)
                logger.info("Start indexing...")
                index_to_doc_store(doc_store, docs, retriever, labels)
                logger.info("Start queries...")

                raw_results = retriever.eval()
                results = {
                    "retriever": retriever_name,
                    "doc_store": doc_store_name,
                    "n_docs": n_docs,
                    "n_queries": raw_results["n_questions"],
                    "retrieve_time": raw_results["retrieve_time"],
                    "queries_per_second": raw_results["n_questions"] / raw_results["retrieve_time"],
                    "seconds_per_query": raw_results["retrieve_time"]/ raw_results["n_questions"],
                    "recall": raw_results["recall"],
                    "map": raw_results["map"],
                    "top_k": raw_results["top_k"],
                    "date_time": datetime.datetime.now(),
                    "error": None
                }

                doc_store.delete_all_documents(index=doc_index)
                doc_store.delete_all_documents(index=label_index)
                time.sleep(5)
                del doc_store
                del retriever
            except Exception as e:
                tb = traceback.format_exc()
                results = {
                    "retriever": retriever_name,
                    "doc_store": doc_store_name,
                    "n_docs": n_docs,
                    "n_queries": 0,
                    "retrieve_time": 0.,
                    "queries_per_second": 0.,
                    "seconds_per_query": 0.,
                    "recall": 0.,
                    "map": 0.,
                    "top_k": 0,
                    "date_time": datetime.datetime.now(),
                    "error": str(tb)
                }
                doc_store.delete_all_documents(index=doc_index)
                doc_store.delete_all_documents(index=label_index)
                time.sleep(5)
                del doc_store
                del retriever
            logger.info(results)
            retriever_results.append(results)

            retriever_df = pd.DataFrame.from_records(retriever_results)
            retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
            retriever_df.to_csv("retriever_query_results.csv")




def add_precomputed_embeddings(embeddings_dir, embeddings_filenames, docs):
    ret = []
    id_to_doc = {x.meta["passage_id"]: x for x in docs}
    for ef in embeddings_filenames:
        logger.info(f"Adding precomputed embeddings from {embeddings_dir + ef}")
        filename = embeddings_dir + ef
        embeds = pickle.load(open(filename, "rb"))
        for i, vec in embeds:
            if int(i) in id_to_doc:
                curr = id_to_doc[int(i)]
                curr.embedding = vec
                ret.append(curr)
    # In the official DPR repo, there are only 20594995 precomputed embeddings for 21015324 wikipedia passages
    # If there isn't an embedding for a given doc, we remove it here
    ret = [x for x in ret if x.embedding is not None]
    logger.info(f"Embeddings loaded for {len(ret)}/{len(docs)} docs")
    return ret


def prepare_data(data_dir, filename_gold, filename_negative, data_s3_url,  embeddings_filenames, embeddings_dir, n_docs=None, n_queries=None, add_precomputed=False):
    """
    filename_gold points to a squad format file.
    filename_negative points to a csv file where the first column is doc_id and second is document text.
    If add_precomputed is True, this fn will look in the embeddings files for precomputed embeddings to add to each Document
    """

    logging.getLogger("farm").setLevel(logging.INFO)
    download_from_s3(data_s3_url + filename_gold, cache_dir=data_dir)
    download_from_s3(data_s3_url + filename_negative, cache_dir=data_dir)
    if add_precomputed:
        for embedding_filename in embeddings_filenames:
            download_from_s3(data_s3_url + str(embeddings_dir) + embedding_filename, cache_dir=data_dir)
    logging.getLogger("farm").setLevel(logging.WARN)

    gold_docs, labels = eval_data_from_file(data_dir + filename_gold)

    # Reduce number of docs
    gold_docs = gold_docs[:n_docs]

    # Remove labels whose gold docs have been removed
    doc_ids = [x.id for x in gold_docs]
    labels = [x for x in labels if x.document_id in doc_ids]

    # Filter labels down to n_queries
    selected_queries = list(set(f"{x.document_id} | {x.question}" for x in labels))
    selected_queries = selected_queries[:n_queries]
    labels = [x for x in labels if f"{x.document_id} | {x.question}" in selected_queries]

    n_neg_docs = max(0, n_docs - len(gold_docs))
    neg_docs = prepare_negative_passages(data_dir, filename_negative, n_neg_docs)
    docs = gold_docs + neg_docs

    if add_precomputed:
        docs = add_precomputed_embeddings(data_dir + embeddings_dir, embeddings_filenames, docs)

    return docs, labels

def prepare_negative_passages(data_dir, filename_negative, n_docs):
    if n_docs == 0:
        return []
    with open(data_dir + filename_negative) as f:
        lines = []
        _ = f.readline() # Skip column titles line
        for _ in range(n_docs):
            lines.append(f.readline()[:-1])

    docs = []
    for l in lines[:n_docs]:
        id, text, title = l.split("\t")
        d = {"text": text,
             "meta": {"passage_id": int(id),
                      "title": title}}
        d = Document(**d)
        docs.append(d)
    return docs


if __name__ == "__main__":
    params, filenames = load_config(config_filename="config.json", ci=True)
    benchmark_indexing(**params, **filenames)
    benchmark_querying(**params, **filenames)
