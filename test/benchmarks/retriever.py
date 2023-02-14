import pandas as pd
from pathlib import Path
from time import perf_counter
from utils import get_document_store, get_retriever, index_to_doc_store, load_config, download_from_url
from haystack.document_stores.utils import eval_data_from_json
from haystack.document_stores.faiss import FAISSDocumentStore

from haystack.schema import Document
import pickle
import time
from tqdm import tqdm
import logging
import datetime
import random
import traceback
import json
from results_to_json import retriever as retriever_json
from templates import RETRIEVER_TEMPLATE, RETRIEVER_MAP_TEMPLATE, RETRIEVER_SPEED_TEMPLATE
from haystack.utils import stop_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("haystack.retriever.base").setLevel(logging.WARN)
logging.getLogger("elasticsearch").setLevel(logging.WARN)

doc_index = "eval_document"
label_index = "label"

index_results_file = "retriever_index_results.csv"
query_results_file = "retriever_query_results.csv"

overview_json = "../../docs/_src/benchmarks/retriever_performance.json"
map_json = "../../docs/_src/benchmarks/retriever_map.json"
speed_json = "../../docs/_src/benchmarks/retriever_speed.json"

DEVICES = None


seed = 42
random.seed(42)


def benchmark_indexing(
    n_docs_options,
    retriever_doc_stores,
    data_dir,
    filename_gold,
    filename_negative,
    data_s3_url,
    embeddings_filenames,
    embeddings_dir,
    update_json,
    save_markdown,
    **kwargs,
):
    retriever_results = []
    for n_docs in n_docs_options:
        for retriever_name, doc_store_name in retriever_doc_stores:
            logger.info("##### Start indexing run: %s, %s, %s docs ##### ", retriever_name, doc_store_name, n_docs)
            try:
                doc_store = get_document_store(doc_store_name)
                retriever = get_retriever(retriever_name, doc_store, DEVICES)
                docs, _ = prepare_data(
                    data_dir=data_dir,
                    filename_gold=filename_gold,
                    filename_negative=filename_negative,
                    remote_url=data_s3_url,
                    embeddings_filenames=embeddings_filenames,
                    embeddings_dir=embeddings_dir,
                    n_docs=n_docs,
                )

                tic = perf_counter()
                index_to_doc_store(doc_store, docs, retriever)
                toc = perf_counter()
                indexing_time = toc - tic

                print(indexing_time)

                retriever_results.append(
                    {
                        "retriever": retriever_name,
                        "doc_store": doc_store_name,
                        "n_docs": n_docs,
                        "indexing_time": indexing_time,
                        "docs_per_second": n_docs / indexing_time,
                        "date_time": datetime.datetime.now(),
                        "error": None,
                    }
                )
                retriever_df = pd.DataFrame.from_records(retriever_results)
                retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
                retriever_df.to_csv(index_results_file)
                logger.info("Deleting all docs from this run ...")

                if isinstance(doc_store, FAISSDocumentStore):
                    doc_store.session.close()
                else:
                    doc_store.delete_documents(index=doc_index)
                    doc_store.delete_documents(index=label_index)

                if save_markdown:
                    md_file = index_results_file.replace(".csv", ".md")
                    with open(md_file, "w") as f:
                        f.write(str(retriever_df.to_markdown()))
                time.sleep(10)
                stop_service(doc_store)
                del doc_store
                del retriever

            except Exception:
                tb = traceback.format_exc()
                logging.error(
                    f"##### The following Error was raised while running indexing run: {retriever_name}, {doc_store_name}, {n_docs} docs #####"
                )
                logging.error(tb)
                retriever_results.append(
                    {
                        "retriever": retriever_name,
                        "doc_store": doc_store_name,
                        "n_docs": n_docs,
                        "indexing_time": 0,
                        "docs_per_second": 0,
                        "date_time": datetime.datetime.now(),
                        "error": str(tb),
                    }
                )
                logger.info("Deleting all docs from this run ...")
                if isinstance(doc_store, FAISSDocumentStore):
                    doc_store.session.close()
                else:
                    doc_store.delete_documents(index=doc_index)
                    doc_store.delete_documents(index=label_index)
                time.sleep(10)
                stop_service(doc_store)
                del doc_store
                del retriever
    if update_json:
        populate_retriever_json()


def benchmark_querying(
    n_docs_options,
    retriever_doc_stores,
    data_dir,
    data_s3_url,
    filename_gold,
    filename_negative,
    n_queries,
    embeddings_filenames,
    embeddings_dir,
    update_json,
    save_markdown,
    wait_write_limit=100,
    **kwargs,
):
    """Benchmark the time it takes to perform querying. Doc embeddings are loaded from file."""
    retriever_results = []

    for n_docs in n_docs_options:
        for retriever_name, doc_store_name in retriever_doc_stores:
            try:
                logger.info("##### Start querying run: %s, %s, %s docs ##### ", retriever_name, doc_store_name, n_docs)
                if retriever_name in ["elastic", "sentence_transformers"]:
                    similarity = "cosine"
                else:
                    similarity = "dot_product"
                doc_store = get_document_store(doc_store_name, similarity=similarity)
                retriever = get_retriever(retriever_name, doc_store, DEVICES)
                add_precomputed = retriever_name in ["dpr"]
                # For DPR, precomputed embeddings are loaded from file
                docs, labels = prepare_data(
                    data_dir=data_dir,
                    filename_gold=filename_gold,
                    filename_negative=filename_negative,
                    remote_url=data_s3_url,
                    embeddings_filenames=embeddings_filenames,
                    embeddings_dir=embeddings_dir,
                    n_docs=n_docs,
                    n_queries=n_queries,
                    add_precomputed=add_precomputed,
                )
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
                    "seconds_per_query": raw_results["retrieve_time"] / raw_results["n_questions"],
                    "recall": raw_results["recall"] * 100,
                    "map": raw_results["map"] * 100,
                    "top_k": raw_results["top_k"],
                    "date_time": datetime.datetime.now(),
                    "error": None,
                }

                logger.info("Deleting all docs from this run ...")
                if isinstance(doc_store, FAISSDocumentStore):
                    doc_store.session.close()
                else:
                    doc_store.delete_documents(index=doc_index)
                    doc_store.delete_documents(index=label_index)
                time.sleep(5)
                stop_service(doc_store)
                del doc_store
                del retriever
            except Exception:
                tb = traceback.format_exc()
                logging.error(
                    f"##### The following Error was raised while running querying run: {retriever_name}, {doc_store_name}, {n_docs} docs #####"
                )
                logging.error(tb)
                results = {
                    "retriever": retriever_name,
                    "doc_store": doc_store_name,
                    "n_docs": n_docs,
                    "n_queries": 0,
                    "retrieve_time": 0.0,
                    "queries_per_second": 0.0,
                    "seconds_per_query": 0.0,
                    "recall": 0.0,
                    "map": 0.0,
                    "top_k": 0,
                    "date_time": datetime.datetime.now(),
                    "error": str(tb),
                }
                logger.info("Deleting all docs from this run ...")
                if isinstance(doc_store, FAISSDocumentStore):
                    doc_store.session.close()
                else:
                    doc_store.delete_documents(index=doc_index)
                    doc_store.delete_documents(index=label_index)
                time.sleep(5)
                del doc_store
                del retriever
            logger.info(results)
            retriever_results.append(results)

            retriever_df = pd.DataFrame.from_records(retriever_results)
            retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
            retriever_df.to_csv(query_results_file)
            if save_markdown:
                md_file = query_results_file.replace(".csv", ".md")
                with open(md_file, "w") as f:
                    f.write(str(retriever_df.to_markdown()))
    if update_json:
        populate_retriever_json()


def populate_retriever_json():
    retriever_overview_data, retriever_map_data, retriever_speed_data = retriever_json(
        index_csv=index_results_file, query_csv=query_results_file
    )
    overview = RETRIEVER_TEMPLATE
    overview["data"] = retriever_overview_data
    map = RETRIEVER_MAP_TEMPLATE
    map["data"] = retriever_map_data
    speed = RETRIEVER_SPEED_TEMPLATE
    speed["data"] = retriever_speed_data
    json.dump(overview, open(overview_json, "w"), indent=4)
    json.dump(speed, open(speed_json, "w"), indent=4)
    json.dump(map, open(map_json, "w"), indent=4)


def add_precomputed_embeddings(embeddings_dir, embeddings_filenames, docs):
    ret = []
    id_to_doc = {x.meta["passage_id"]: x for x in docs}
    for ef in embeddings_filenames:
        logger.info("Adding precomputed embeddings from %s", embeddings_dir + ef)
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
    logger.info("Embeddings loaded for %s/%s docs", len(ret), len(docs))
    return ret


def prepare_data(
    data_dir,
    filename_gold,
    filename_negative,
    remote_url,
    embeddings_filenames,
    embeddings_dir,
    n_docs=None,
    n_queries=None,
    add_precomputed=False,
):
    """
    filename_gold points to a squad format file.
    filename_negative points to a csv file where the first column is doc_id and second is document text.
    If add_precomputed is True, this fn will look in the embeddings files for precomputed embeddings to add to each Document
    """

    logging.getLogger("farm").setLevel(logging.INFO)
    download_from_url(remote_url + filename_gold, filepath=data_dir + filename_gold)
    download_from_url(remote_url + filename_negative, filepath=data_dir + filename_negative)
    if add_precomputed:
        for embedding_filename in embeddings_filenames:
            download_from_url(
                remote_url + str(embeddings_dir) + embedding_filename,
                filepath=data_dir + str(embeddings_dir) + embedding_filename,
            )
    logging.getLogger("farm").setLevel(logging.WARN)

    gold_docs, labels = eval_data_from_json(data_dir + filename_gold)

    # Reduce number of docs
    gold_docs = gold_docs[:n_docs]

    # Remove labels whose gold docs have been removed
    doc_ids = [x.id for x in gold_docs]
    labels = [x for x in labels if x.document.id in doc_ids]

    # Filter labels down to n_queries
    selected_queries = list(set(f"{x.document.id} | {x.query}" for x in labels))
    selected_queries = selected_queries[:n_queries]
    labels = [x for x in labels if f"{x.document.id} | {x.query}" in selected_queries]

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
        _ = f.readline()  # Skip column titles line
        for _ in range(n_docs):
            lines.append(f.readline()[:-1])

    docs = []
    for l in lines[:n_docs]:
        id, text, title = l.split("\t")
        d = {"content": text, "meta": {"passage_id": int(id), "title": title}}
        d = Document(**d)
        docs.append(d)
    return docs


if __name__ == "__main__":
    params, filenames = load_config(config_filename="config.json", ci=True)
    benchmark_indexing(**params, **filenames, update_json=True, save_markdown=False)
    benchmark_querying(**params, **filenames, update_json=True, save_markdown=False)
