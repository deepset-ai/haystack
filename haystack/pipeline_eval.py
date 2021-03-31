from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.utils import fetch_archive_from_http
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from haystack import Pipeline
from farm.utils import initialize_device_settings
from haystack.preprocessor import PreProcessor
from haystack.eval import EvalReader, EvalRetriever

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

LAUNCH_ELASTICSEARCH = True
doc_index = "documents"
label_index = "labels"
top_k_retriever = 10
open_domain = False

def launch_es():
    logger.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'], shell=True
    )
    if status.returncode:
        logger.warning("Tried to start Elasticsearch through Docker but this failed. "
                       "It is likely that there is already an existing Elasticsearch instance running. ")
    else:
        time.sleep(15)

def main():

    launch_es()

    document_store = ElasticsearchDocumentStore()
    es_retriever = ElasticsearchRetriever(document_store=document_store)
    eval_retriever = EvalRetriever(open_domain=open_domain)
    reader = FARMReader("deepset/roberta-base-squad2", top_k_per_candidate=4, num_processes=1, return_no_answer=True)
    eval_reader = EvalReader(debug=True, open_domain=open_domain)

    # Download evaluation data, which is a subset of Natural Questions development set containing 50 documents
    doc_dir = "../data/nq"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Add evaluation data to Elasticsearch document store
    # We first delete the custom tutorial indices to not have duplicate elements
    preprocessor = PreProcessor(split_length=500, split_overlap=0, split_respect_sentence_boundary=False, clean_empty_lines=False, clean_whitespace=False)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(
        filename="../data/nq/nq_dev_subset_v2.json",
        doc_index=doc_index,
        label_index=label_index,
        preprocessor=preprocessor
    )
    labels = document_store.get_all_labels_aggregated(index=label_index)
    q_to_l_dict = {
        l.question: {
            "retriever": l,
            "reader": l
        } for l in labels
    }

    # Here is the pipeline definition
    p = Pipeline()
    p.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["ESRetriever"])
    p.add_node(component=reader, name="QAReader", inputs=["EvalRetriever"])
    p.add_node(component=eval_reader, name="EvalReader", inputs=["QAReader"])

    results = []
    for i, (q, l) in enumerate(q_to_l_dict.items()):
        res = p.run(query=q,
                    top_k_retriever=top_k_retriever,
                    labels=l,
                    top_k_reader=10,
                    index=doc_index,
                    # skip_incorrect_retrieval=True
                    )
        results.append(res)

    eval_retriever.print()
    print()
    es_retriever.print_time()
    print()
    eval_reader.print(mode="reader")
    print()
    reader.print_time()
    print()
    eval_reader.print(mode="pipeline")



if __name__ == "__main__":
    main()