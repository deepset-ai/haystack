import json
import logging
import subprocess
import time

from tqdm import tqdm

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store import MilvusDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers, launch_es, launch_milvus
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from pathlib import Path
from haystack.preprocessor import PreProcessor

def kendra_benchmark():
    # Define variables
    logger = logging.getLogger(__name__)
    doc_dir = Path("../data/nq")
    doc_index = "document"
    outfile = "predictions.json"
    summary_file = "summary.json"
    model = "ankur310794/roberta-base-squad2-nq"
    top_k_retriever = 5
    top_k_reader = 10
    split_length = 100
    split_overlap = 10
    dense_retrieval = True

    if not dense_retrieval:
        launch_es()
        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index=doc_index)
    else:
        launch_milvus()
        document_store = MilvusDocumentStore()

    document_store.delete_documents(index=doc_index)
    preprocessor = PreProcessor(
        clean_header_footer=True,
        split_length=split_length,
        split_overlap=split_overlap,
        split_by="word",
        split_respect_sentence_boundary=False
    )

    def prepare_docs(filepath, preprocessor):
        lines = json.load(open(filepath))
        print(len(lines))
        lines = set(lines)
        print(len(lines))
        docs = [{"text": l} for l in lines]
        docs = preprocessor.process(docs)
        print(len(docs))
        return docs

    if not dense_retrieval:
        retriever = ElasticsearchRetriever(document_store=document_store, top_k=top_k_retriever)
    else:
        retriever = DensePassageRetriever(document_store=document_store, top_k=top_k_retriever)

    docs = prepare_docs(doc_dir / "nq_dev_texts.json", preprocessor)
    tic = time.perf_counter()
    document_store.write_documents(docs, index=doc_index)
    toc = time.perf_counter()
    index_time = toc - tic

    if dense_retrieval:
        tic = time.perf_counter()
        document_store.update_embeddings(retriever=retriever, index=doc_index)
        toc = time.perf_counter()
        embedding_update_time = toc - tic



    reader = FARMReader(model_name_or_path=model, use_gpu=True, top_k=top_k_reader)

    from haystack.pipeline import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    label_data = json.load(open(doc_dir / "nq_dev_labels.json"))
    questions = [x["question"] for x in label_data]

    # querying
    results = []
    tic = time.perf_counter()
    for i, q in tqdm(enumerate(questions)):
        result = pipe.run(query=q)
        results.append(result)
    toc = time.perf_counter()
    query_time = toc - tic
    n_docs = len(questions)
    summary = {
        "query_time": query_time,
        "index_time": index_time,
        "n_docs": n_docs,
        "top_k_reader": top_k_reader,
        "top_k_retriever": top_k_retriever,
        "reader": str(type(reader)),
        "reader_model": model,
        "retriever": str(type(retriever)),
        "split_length": split_length
    }
    if dense_retrieval:
        summary["embedding_update_time"] = embedding_update_time
    json.dump(summary, open(doc_dir / summary_file, "w"), indent=2)
    json.dump(results, open(doc_dir / outfile, "w"), indent=2)


    ## Voilà! Ask a question!
    # prediction = pipe.run(query="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)
    #
    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", top_k_reader=5)
    # prediction = pipe.run(query="Who is the sister of Sansa?", top_k_reader=5)
    #
    # print_answers(prediction, details="minimal")


if __name__ == "__main__":
    kendra_benchmark()
