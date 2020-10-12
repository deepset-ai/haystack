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
        "f1": "F1",
        "passages_per_second": "Speed",
        "reader": "Model"
    }

    df = pd.read_csv("reader_results.csv")
    df = df[["f1", "passages_per_second", "reader"]]
    df["reader"] = df["reader"].map(model_rename_map)
    df = df[list(column_name_map)]
    df = df.rename(columns=column_name_map)
    ret = [dict(row) for i, row in df.iterrows()]
    print("Reader overview")
    print(json.dumps(ret, indent=2))

def retriever():


    column_name_map = {
        "model": "model",
        "n_docs": "n_docs",
        "docs_per_second": "index_speed",
        "queries_per_second": "query_speed",
        "map": "map"
    }

    name_cleaning = {
        "dpr": "DPR",
        "elastic": "BM25",
        "elasticsearch": "ElasticSearch",
        "faiss": "FAISS",
        "faiss_flat": "FAISS (flat)",
        "faiss_hnsw": "FAISS (HSNW)"
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

    print("Retriever overview")
    print(retriever_overview(df))

    print("Retriever MAP")
    print(retriever_map(df))

    print("Retriever Speed")
    print(retriever_speed(df))


def retriever_map(df):
    columns = ["model", "n_docs", "map"]
    df = df[columns]
    ret = [list(row) for i, row in df.iterrows()]
    ret = [columns] + ret
    return json.dumps(ret, indent=4)


def retriever_speed(df):
    columns = ["model", "n_docs", "query_speed"]
    df = df[columns]
    ret = [list(row) for i, row in df.iterrows()]
    ret = [columns] + ret
    return json.dumps(ret, indent=4)



def retriever_overview(df, chosen_n_docs=100_000):

    df = df[df["n_docs"] == chosen_n_docs]
    ret = [dict(row) for i, row in df.iterrows()]

    return json.dumps(ret, indent=2)


if __name__ == "__main__":
    reader()
    retriever()