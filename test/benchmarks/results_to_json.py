import json
import pandas as pd
from pprint import pprint

def reader():

    model_rename_map = {
        'deepset/roberta-base-squad2': "RoBERTa",
        'deepset/minilm-uncased-squad2': "MiniLM",
        'deepset/bert-base-cased-squad2': "BERT base",
        'deepset/bert-large-uncased-whole-word-masking-squad2': "BERT large",
        'deepset/xlm-roberta-large-squad2': "XLM-RoBERTa",
    }

    column_name_map = {
        "f1": "f1",
        "passages_per_second": "speed",
        "reader": "model"
    }

    df = pd.read_csv("reader_results.csv")
    df = df[["f1", "passages_per_second", "reader"]]
    df["reader"] = df["reader"].map(model_rename_map)
    df = df[list(column_name_map)]
    df = df.rename(columns=column_name_map)
    ret = [dict(row) for i, row in df.iterrows()]
    pprint(ret)

def retriever():

    column_name_map = {
        "model": "model",
        "n_docs": "n_docs",
        "docs_per_second": "index_sped",
        "queries_per_second": "querying_speed",
        "recall": "recall"
    }

    name_cleaning = {
        "dpr": "DPR",
        "elastic": "BM25",
        "elasticsearch": "ElasticSearch",
        "faiss": "FAISS"
    }

    index = pd.read_csv("retriever_index_results.csv")
    query = pd.read_csv("retriever_query_results.csv")
    df = pd.merge(index, query,
                  how="right",
                  left_on=["retriever", "doc_store", "n_docs"],
                  right_on=["retriever", "doc_store", "n_docs"])
    df["retriever"] = df["retriever"].map(name_cleaning)
    df["doc_store"] = df["doc_store"].map(name_cleaning)
    df["model"] = df["retriever"] + " / " + df["doc_store"]
    df = df[list(column_name_map)]
    df = df.rename(columns=column_name_map)
    ret = [dict(row) for i, row in df.iterrows()]
    pprint(ret)

if __name__ == "__main__":
    reader()
    retriever()