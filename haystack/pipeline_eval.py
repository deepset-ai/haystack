
from farm.evaluation.squad_evaluation import compute_f1 as calculate_f1_str
from farm.evaluation.squad_evaluation import compute_exact as calculate_em_str
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.utils import fetch_archive_from_http
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from haystack.finder import Finder
from haystack import Pipeline
from farm.utils import initialize_device_settings
from haystack.preprocessor import PreProcessor

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

LAUNCH_ELASTICSEARCH = True
doc_index = "tutorial5_docs"
label_index = "tutorial5_labels"
top_k_retriever = 10

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
    eval_retriever = EvalRetriever()
    reader = FARMReader("deepset/roberta-base-squad2", top_k_per_candidate=4, num_processes=1, return_no_answer=True)
    eval_reader = EvalReader(debug=True)

    # Download evaluation data, which is a subset of Natural Questions development set containing 50 documents
    doc_dir = "../data/nq"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Add evaluation data to Elasticsearch document store
    # We first delete the custom tutorial indices to not have duplicate elements
    preprocessor = PreProcessor(split_length=500, split_overlap=0, split_respect_sentence_boundary=False, clean_empty_lines=False, clean_whitespace=False)
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(filename="../data/nq/nq_dev_subset_v2.json", doc_index=doc_index, label_index=label_index, preprocessor=preprocessor)
    labels = document_store.get_all_labels(index=label_index)
    q_to_l_dict = {x.question: x.answer for x in labels}

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


class EvalRetriever:
    def __init__(self, debug=False):
        self.outgoing_edges = 1
        self.correct_retrieval_count = 0
        self.query_count = 0
        self.has_answer_count = 0
        self.has_answer_correct = 0
        self.has_answer_recall = 0
        self.no_answer_count = 0
        self.recall = 0.0
        self.no_answer_warning = False
        self.debug = debug
        self.log = []

    def run(self, documents, labels, **kwargs):
        # Open domain mode
        self.query_count += 1
        if type(labels) == str:
            labels = [labels]
        texts = [x.text for x in documents]
        correct_retrieval = 0
        if "" in labels:
            self.no_answer_count += 1
            correct_retrieval = 1
            if not self.no_answer_warning:
                self.no_answer_warning = True
                logger.warning("There seem to be empty string labels in the dataset suggesting that there "
                               "are samples with is_impossible=True. "
                               "Retrieval of these samples is always treated as correct.")
        else:
            self.has_answer_count += 1
            for t in texts:
                for label in labels:
                    if label.lower() in t.lower():
                        correct_retrieval = 1
                        self.has_answer_correct += 1
                        break
                if correct_retrieval:
                    break
        self.correct_retrieval_count += correct_retrieval
        self.recall = self.correct_retrieval_count / self.query_count
        self.has_answer_recall = self.has_answer_correct / self.has_answer_count
        if self.debug:
            self.log.append({"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs})
        return {"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs}, "output_1"

    def print(self):
        print("Retriever")
        print("-----------------")
        if self.no_answer_count:
            print(
                f"has_answer recall: {self.has_answer_recall} ({self.has_answer_correct}/{self.has_answer_count})")
            print(
                f"no_answer recall:  1.00 ({self.no_answer_count}/{self.no_answer_count}) (no_answer samples are always treated as correctly retrieved)")
        print(f"recall: {self.recall} ({self.correct_retrieval_count} / {self.query_count})")


class EvalReader:
    def __init__(self, debug=False, skip_incorrect_retrieval=True):
        self.outgoing_edges = 1
        self.query_count = 0
        self.correct_retrieval_count = 0
        self.no_answer_count = 0
        self.has_answer_count = 0
        self.top_1_no_answer_count = 0
        self.top_1_em_count = 0
        self.top_k_em_count = 0
        self.top_1_f1_sum = 0
        self.top_k_f1_sum = 0
        self.top_1_no_answer = 0
        self.top_1_em = 0.0
        self.top_k_em = 0.0
        self.top_1_f1 = 0.0
        self.top_k_f1 = 0.0
        self.log = []
        self.debug = debug
        self.skip_incorrect_retrieval = skip_incorrect_retrieval

    def run(self, **kwargs):
        self.query_count += 1
        predictions = [p["answer"] for p in kwargs["answers"]]
        predictions = [x if x else "" for x in predictions]
        skip = self.skip_incorrect_retrieval and not kwargs.get("correct_retrieval")
        if predictions and not skip:
            self.correct_retrieval_count += 1
            gold_labels = kwargs["labels"]
            if "" in gold_labels:
                self.no_answer_count += 1
                if predictions[0] == "":
                    self.top_1_no_answer_count += 1
                if self.debug:
                    self.log.append({"predictions": predictions,
                                     "gold_labels": gold_labels,
                                     "top_1_no_answer": int(predictions[0] == ""),
                                     })
                self.update_no_answer_metrics()

            else:
                self.has_answer_count += 1
                curr_top_1_em = calculate_em_str_multi(gold_labels, predictions[0])
                curr_top_1_f1 = calculate_f1_str_multi(gold_labels, predictions[0])
                curr_top_k_em = max([calculate_em_str_multi(gold_labels, p) for p in predictions])
                curr_top_k_f1 = max([calculate_f1_str_multi(gold_labels, p) for p in predictions])

                self.top_1_em_count += curr_top_1_em
                self.top_1_f1_sum += curr_top_1_f1
                self.top_k_em_count += curr_top_k_em
                self.top_k_f1_sum += curr_top_k_f1
                if self.debug:
                    self.log.append({"predictions": predictions,
                                     "gold_labels": gold_labels,
                                     "top_k_f1": curr_top_k_f1,
                                     "top_k_em": curr_top_k_em
                                     })
                self.update_has_answer_metrics()

        return {**kwargs}, "output_1"

    def update_has_answer_metrics(self):
        self.top_1_em = self.top_1_em_count / self.has_answer_count
        self.top_k_em = self.top_k_em_count / self.has_answer_count
        self.top_1_f1 = self.top_1_f1_sum / self.has_answer_count
        self.top_k_f1 = self.top_k_f1_sum / self.has_answer_count

    def update_no_answer_metrics(self):
        self.top_1_no_answer = self.top_1_no_answer_count / self.no_answer_count

    def print(self, mode):
        if mode == "reader":
            print("Reader")
            print("-----------------")
            # print(f"answer in retrieved docs: {correct_retrieval}")
            print(f"has answer queries: {self.has_answer_count}")
            print(f"top 1 EM: {self.top_1_em}")
            print(f"top k EM: {self.top_k_em}")
            print(f"top 1 F1: {self.top_1_f1}")
            print(f"top k F1: {self.top_k_f1}")
            if self.no_answer_count:
                print()
                print(f"no_answer queries: {self.no_answer_count}")
                print(f"top 1 no_answer accuracy: {self.top_1_no_answer}")
        elif mode == "pipeline":
            print("Pipeline")
            print("-----------------")

            pipeline_top_1_em = (self.top_1_em_count + self.top_1_no_answer_count) / self.query_count
            pipeline_top_k_em = (self.top_k_em_count + self.no_answer_count) / self.query_count
            pipeline_top_1_f1 = (self.top_1_f1_sum + self.top_1_no_answer_count) / self.query_count
            pipeline_top_k_f1 = (self.top_k_f1_sum + self.no_answer_count) / self.query_count

            print(f"queries: {self.query_count}")
            print(f"top 1 EM: {pipeline_top_1_em}")
            print(f"top k EM: {pipeline_top_k_em}")
            print(f"top 1 F1: {pipeline_top_1_f1}")
            print(f"top k F1: {pipeline_top_k_f1}")
            if self.no_answer_count:
                print(
                    "(top k results are likely inflated since the Reader always returns a no_answer prediction in its top k)"
                )


def calculate_em_str_multi(gold_labels, prediction):
    for gold_label in gold_labels:
        result = calculate_em_str(gold_label, prediction)
        if result == 1.0:
            return 1.0
    return 0.0

def calculate_f1_str_multi(gold_labels, prediction):
    results = []
    for gold_label in gold_labels:
        result = calculate_f1_str(gold_label, prediction)
        results.append(result)
    return max(results)

if __name__ == "__main__":
    main()