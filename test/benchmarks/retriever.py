import pandas as pd
from pathlib import Path
from time import perf_counter
from utils import get_document_store, get_retriever
from haystack.preprocessor.utils import eval_data_from_file


retriever_doc_stores = [("dpr", "faiss"), ("elastic", "elasticsearch")]
n_docs_options = [1000]

data_dir = Path("../../data/retriever")
filename_gold = "nq2squad-dev.json"            # Found at s3://ext-haystack-retriever-eval
filename_negative = "psgs_w100_minus_gold.tsv"      # Found at s3://ext-haystack-retriever-eval

doc_index = "eval_document"
label_index = "label"


def benchmark_speed():
    retriever_results = []
    for retriever_name, doc_store_name in retriever_doc_stores:
        for n_docs in n_docs_options:
            # try:
            doc_store = get_document_store(doc_store_name)
            retriever = get_retriever(retriever_name, doc_store)
            docs, labels = prepare_data(data_dir, filename_gold, filename_negative, n_docs=n_docs)

            doc_store, indexing_time = benchmark_indexing_speed(doc_store, docs, labels, retriever)
            print(indexing_time)

            results = retriever.eval()
            # results["indexing_time"] = indexing_time
            # results["retriever"] = retriever_name
            # results["doc_store"] = doc_store_name
            # print(results)
            # retriever_results.append(results)
            # except Exception as e:
            #     retriever_results.append(str(e))

            retriever_df = pd.DataFrame.from_records(retriever_results)
            retriever_df.to_csv("retriever_results.csv")

def prepare_data(data_dir, filename_gold, filename_negative, n_docs=None):
    """
    filename_gold points to a squad format file.
    filename_negative points to a csv file where the first column is doc_id and second is document text.
    """
    gold_docs, labels = eval_data_from_file(data_dir / filename_gold)
    gold_docs = gold_docs[:n_docs]
    n_neg_docs = max(0, n_docs - len(gold_docs))
    neg_docs = prepare_negative_passages(data_dir, filename_negative, n_neg_docs)
    docs = gold_docs + neg_docs
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
             "meta": {"id": id,
                      "title": title}}
        docs.append(d)
    return docs

def benchmark_indexing_speed(doc_store, docs, labels, retriever):
    tic = perf_counter()
    index_to_doc_store(doc_store, docs, labels, retriever)
    toc = perf_counter()
    time = toc - tic
    return doc_store, time

def index_to_doc_store(doc_store, docs, labels, retriever):
    doc_store.delete_all_documents(index=doc_index)
    doc_store.delete_all_documents(index=label_index)
    doc_store.write_documents(docs, doc_index)
    doc_store.write_labels(labels, index=label_index)
    if callable(getattr(retriever, "embed_passages", None)):
        doc_store.update_embeddings(retriever, index=doc_index)
    else:
        pass


if __name__ == "__main__":
    benchmark_speed()