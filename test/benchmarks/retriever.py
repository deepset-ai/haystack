import pandas as pd
from pathlib import Path
from time import perf_counter
from utils import get_document_store, get_retriever, index_to_doc_store
from haystack.preprocessor.utils import eval_data_from_file
from haystack import Document
import pickle
from tqdm import tqdm
import logging
import datetime
import random


logger = logging.getLogger(__name__)
logging.getLogger("haystack.retriever.base").setLevel(logging.WARN)
logging.getLogger("elasticsearch").setLevel(logging.WARN)

retriever_doc_stores = [
    # ("elastic", "elasticsearch"),
    # ("dpr", "elasticsearch"),
    ("dpr", "faiss_flat"),
    # ("dpr", "faiss_hnsw")
]

n_docs_options = [
    1000,
    10000,
    100000,
    500000
]

# If set to None, querying will be run on all queries
n_queries = 100
# shuffling of neg. passages slows the runs down but makes accuracy benchmarks more reliable
shuffle_negatives = True
data_dir = Path("../../data/retriever")
filename_gold = "nq2squad-dev.json"            # Found at s3://ext-haystack-retriever-eval
filename_negative = "psgs_w100_minus_gold.tsv"      # Found at s3://ext-haystack-retriever-eval
embeddings_dir = Path("embeddings")
embeddings_filenames = [f"wikipedia_passages_1m.pkl"]   # Found at s3://ext-haystack-retriever-eval

doc_index = "eval_document"
label_index = "label"

seed = 42

random.seed(42)


def prepare_data(data_dir, filename_gold, filename_negative, n_docs=None, n_queries=None, add_precomputed=False, shuffle_negatives=False):
    """
    filename_gold points to a squad format file.
    filename_negative points to a csv file where the first column is doc_id and second is document text.
    If add_precomputed is True, this fn will look in the embeddings files for precomputed embeddings to add to each Document
    """

    gold_docs, labels = eval_data_from_file(data_dir / filename_gold)

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
    neg_docs = prepare_negative_passages(data_dir, filename_negative, n_neg_docs, shuffle_negatives)
    docs = gold_docs + neg_docs

    if add_precomputed:
        docs = add_precomputed_embeddings(data_dir / embeddings_dir, embeddings_filenames, docs)

    return docs, labels

def prepare_negative_passages(data_dir, filename_negative, n_docs, shuffle=False):
    if n_docs == 0:
        return []
    with open(data_dir / filename_negative) as f:
        if shuffle:
            lines = [l[:-1] for l in f][1:]     # Skip column titles line
            random.shuffle(lines)
        else:
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

def benchmark_indexing():

    retriever_results = []
    for n_docs in n_docs_options:
        for retriever_name, doc_store_name in retriever_doc_stores:
            doc_store = get_document_store(doc_store_name)
            retriever = get_retriever(retriever_name, doc_store)

            docs, _ = prepare_data(data_dir, filename_gold, filename_negative, n_docs=n_docs,
                                   shuffle_negatives=shuffle_negatives)

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
                "date_time": datetime.datetime.now()})
            retriever_df = pd.DataFrame.from_records(retriever_results)
            retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
            retriever_df.to_csv("retriever_index_results.csv")

            del doc_store
            del retriever

def benchmark_querying():
    """ Benchmark the time it takes to perform querying. Doc embeddings are loaded from file."""
    retriever_results = []
    for n_docs in n_docs_options:
        for retriever_name, doc_store_name in retriever_doc_stores:
            logger.info(f"##### Start run: {retriever_name}, {doc_store_name}, {n_docs} docs ##### ")
            doc_store = get_document_store(doc_store_name)
            retriever = get_retriever(retriever_name, doc_store)
            add_precomputed = retriever_name in ["dpr"]
            # For DPR, precomputed embeddings are loaded from file
            docs, labels = prepare_data(data_dir,
                                        filename_gold,
                                        filename_negative,
                                        n_docs=n_docs,
                                        n_queries=n_queries,
                                        add_precomputed=add_precomputed,
                                        shuffle_negatives=shuffle_negatives)
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
                "date_time": datetime.datetime.now()
            }
            logger.info(results)
            retriever_results.append(results)
            del doc_store
            del retriever
            retriever_df = pd.DataFrame.from_records(retriever_results)
            retriever_df = retriever_df.sort_values(by="retriever").sort_values(by="doc_store")
            retriever_df.to_csv("retriever_query_results.csv")




def add_precomputed_embeddings(embeddings_dir, embeddings_filenames, docs):
    ret = []
    id_to_doc = {x.meta["passage_id"]: x for x in docs}
    for ef in embeddings_filenames:
        logger.info(f"Adding precomputed embeddings from {embeddings_dir / ef}")
        filename = embeddings_dir / ef
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


if __name__ == "__main__":
    # benchmark_indexing()
    benchmark_querying()
