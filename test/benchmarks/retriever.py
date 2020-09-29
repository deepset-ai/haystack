import pandas as pd
from pathlib import Path
from time import perf_counter
from utils import get_document_store, get_retriever, index_to_doc_store
from haystack.preprocessor.utils import eval_data_from_file
from haystack import Document
import pickle
from tqdm import tqdm


retriever_doc_stores = [("dpr", "faiss"), ("elastic", "elasticsearch")]
n_docs_options = [1000, 5000, 10000]

data_dir = Path("../../data/retriever")
filename_gold = "nq2squad-dev.json"            # Found at s3://ext-haystack-retriever-eval
filename_negative = "psgs_w100_minus_gold.tsv"      # Found at s3://ext-haystack-retriever-eval
embeddings_dir = Path("embeddings")
embeddings_filenames = [f"wikipedia_passages_1m.pkl"]

doc_index = "eval_document"
label_index = "label"


def benchmark_speed():
    # benchmark_indexing_speed()
    benchmark_querying_speed()


def prepare_data(data_dir, filename_gold, filename_negative, n_docs=None, add_precomputed=False):
    """
    filename_gold points to a squad format file.
    filename_negative points to a csv file where the first column is doc_id and second is document text.
    If add_precomputed is True, this fn will look in the embeddings files for precomputed embeddings to add to each Document
    """

    gold_docs, labels = eval_data_from_file(data_dir / filename_gold)

    # Reduce number of docs and remove labels whose gold docs have been removed
    gold_docs = gold_docs[:n_docs]
    doc_ids = [x.id for x in gold_docs]
    labels = [x for x in labels if x.document_id in doc_ids]

    n_neg_docs = max(0, n_docs - len(gold_docs))
    neg_docs = prepare_negative_passages(data_dir, filename_negative, n_neg_docs)
    docs = gold_docs + neg_docs

    if add_precomputed:
        docs = add_precomputed_embeddings(data_dir / embeddings_dir, embeddings_filenames, docs)
        # In the official DPR repo, there are only 20594995 precomputed embeddings for 21015324 wikipedia passages
        # If there isn't an embedding for a given doc, we remove it here
        docs = [x for x in docs if x.embedding is not None]

    return docs, labels

def prepare_negative_passages(data_dir, filename_negative, n_docs):
    if n_docs == 0:
        return []
    with open(data_dir / filename_negative) as f:
        _ = f.readline()    # skip column titles line
        if not n_docs:
            lines = [l[:-1] for l in f][1:]
        else:
            lines = []
            for _ in range(n_docs):
                lines.append(f.readline()[:-1])
    docs = []
    for l in lines:
        id, text, title = l.split("\t")
        d = {"text": text,
             "meta": {"passage_id": int(id),
                      "title": title}}
        d = Document(**d)
        docs.append(d)
    return docs

def benchmark_indexing_speed():

    retriever_results = []
    for retriever_name, doc_store_name in retriever_doc_stores:
        for n_docs in n_docs_options:
            # try:

            doc_store = get_document_store(doc_store_name)
            retriever = get_retriever(retriever_name, doc_store)

            docs, _ = prepare_data(data_dir, filename_gold, filename_negative, n_docs=n_docs)

            tic = perf_counter()
            index_to_doc_store(doc_store, docs, retriever)
            toc = perf_counter()
            indexing_time = toc - tic

            print(indexing_time)

            retriever_results.append({
                "retriever": retriever_name,
                "doc_store": doc_store_name,
                "n_docs": n_docs,
                "indexing_time": indexing_time})
            retriever_df = pd.DataFrame.from_records(retriever_results)
            retriever_df.to_csv("retriever_index_results.csv")

            del doc_store
            del retriever

def benchmark_querying_speed():
    """ Benchmark the time it takes to perform querying. Doc embeddings are loaded from file."""
    retriever_results = []
    for retriever_name, doc_store_name in retriever_doc_stores:

        for n_docs in n_docs_options:
            doc_store = get_document_store(doc_store_name)
            retriever = get_retriever(retriever_name, doc_store)
            # try:
            add_precomputed = retriever_name in ["dpr"]
            docs, labels = prepare_data(data_dir, filename_gold, filename_negative, n_docs=n_docs, add_precomputed=add_precomputed)
            tic = perf_counter()
            index_to_doc_store(doc_store, docs, retriever, labels)
            toc = perf_counter()
            results = retriever.eval()
            retriever_results.append(results)
            retriever_df = pd.DataFrame.from_records(retriever_results)
            retriever_df.to_csv("retriever_query_results.csv")
            print("index precomputed")
            print(toc- tic)
            del doc_store
            del retriever

def add_precomputed_embeddings(embeddings_dir, embeddings_filenames, docs):
    ret = []
    id_to_doc = {x.meta["passage_id"]: x for x in docs}
    for ef in embeddings_filenames:
        filename = embeddings_dir / ef
        print(filename)
        embeds = pickle.load(open(filename, "rb"))
        for i, vec in tqdm(embeds):
            if int(i) in id_to_doc:
                curr = id_to_doc[int(i)]
                curr.embedding = vec
                ret.append(curr)
    return ret




if __name__ == "__main__":
    benchmark_speed()