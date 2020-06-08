from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import fetch_archive_from_http
from haystack.retriever.elasticsearch import ElasticsearchRetriever
from haystack.reader.farm import FARMReader
from haystack.finder import Finder
from farm.utils import initialize_device_settings

import logging
import subprocess
import time

LAUNCH_ELASTICSEARCH = False
device, n_gpu = initialize_device_settings(use_cuda=True)

# Start an Elasticsearch server
# You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
# your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.
if LAUNCH_ELASTICSEARCH:
    logging.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
    )
    if status.returncode:
        raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                        "then set LAUNCH_ELASTICSEARCH in the script to False.")
    time.sleep(30)

# Download evaluation data, which is a subset of Natural Questions development set containing 50 documents
doc_dir = "../data/nq"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset.json.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document", create_index=False)
# Add evaluation data to Elasticsearch database
document_store.add_eval_data("../data/nq/nq_dev_subset.json")

# Initialize Retriever
retriever = ElasticsearchRetriever(document_store=document_store)

# Initialize Reader
reader = FARMReader("deepset/roberta-base-squad2")

# Initialize Finder which sticks together Reader and Retriever
finder = Finder(reader, retriever)

# Evaluate Retriever on its own
retriever_eval_results = retriever.eval()
## Retriever Recall is the proportion of questions for which the correct document containing the answer is
## among the correct documents
print("Retriever Recall:", retriever_eval_results["recall"])
## Retriever Mean Avg Precision rewards retrievers that give relevant documents a higher rank
print("Retriever Mean Avg Precision:", retriever_eval_results["mean avg precision"])

# Evaluate Reader on its own
reader_eval_results = reader.eval(document_store=document_store, device=device)
# Evaluation of Reader can also be done directly on a SQuAD-formatted file without passing the data to Elasticsearch
#reader_eval_results = reader.eval_on_file("../data/natural_questions", "dev_subset.json", device=device)

## Reader Top-N-Recall is the proportion of predicted answers that overlap with their corresponding correct answer
print("Reader Top-N-Recall:", reader_eval_results["top_n_recall"])
## Reader Exact Match is the proportion of questions where the predicted answer is exactly the same as the correct answer
print("Reader Exact Match:", reader_eval_results["EM"])
## Reader F1-Score is the average overlap between the predicted answers and the correct answers
print("Reader F1-Score:", reader_eval_results["f1"])


# Evaluate combination of Reader and Retriever through Finder
finder_eval_results = finder.eval()

print("\n___Retriever Metrics in Finder___")
print("Retriever Recall:", finder_eval_results["retriever_recall"])
print("Retriever Mean Avg Precision:", finder_eval_results["retriever_map"])

# Reader is only evaluated with those questions, where the correct document is among the retrieved ones
print("\n___Reader Metrics in Finder___")
print("Reader Top-1 accuracy:", finder_eval_results["reader_top1_accuracy"])
print("Reader Top-1 accuracy (has answer):", finder_eval_results["reader_top1_accuracy_has_answer"])
print("Reader Top-k accuracy:", finder_eval_results["reader_top_k_accuracy"])
print("Reader Top-k accuracy (has answer):", finder_eval_results["reader_topk_accuracy_has_answer"])
print("Reader Top-1 EM:", finder_eval_results["reader_top1_em"])
print("Reader Top-1 EM (has answer):", finder_eval_results["reader_top1_em_has_answer"])
print("Reader Top-k EM:", finder_eval_results["reader_topk_em"])
print("Reader Top-k EM (has answer):", finder_eval_results["reader_topk_em_has_answer"])
print("Reader Top-1 F1:", finder_eval_results["reader_top1_f1"])
print("Reader Top-1 F1 (has answer):", finder_eval_results["reader_top1_f1_has_answer"])
print("Reader Top-k F1:", finder_eval_results["reader_topk_f1"])
print("Reader Top-k F1 (has answer):", finder_eval_results["reader_topk_f1_has_answer"])
print("Reader Top-1 no-answer accuracy:", finder_eval_results["reader_top1_no_answer_accuracy"])
print("Reader Top-k no-answer accuracy:", finder_eval_results["reader_topk_no_answer_accuracy"])

# Time measurements
print("\n___Time Measurements___")
print("Total retrieve time:", finder_eval_results["total_retrieve_time"])
print("Avg retrieve time per question:", finder_eval_results["avg_retrieve_time"])
print("Total reader timer:", finder_eval_results["total_reader_time"])
print("Avg read time per question:", finder_eval_results["avg_reader_time"])
print("Total Finder time:", finder_eval_results["total_finder_time"])