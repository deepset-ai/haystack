from utils import get_document_store, index_to_doc_store, get_reader
from haystack.preprocessor.utils import eval_data_from_json
from farm.data_handler.utils import _download_extract_downstream_data

from pathlib import Path
import pandas as pd
from results_to_json import reader as reader_json
from templates import READER_TEMPLATE
import json
import logging

logger = logging.getLogger(__name__)

reader_models_full = ["deepset/roberta-base-squad2", "deepset/minilm-uncased-squad2",
                 "deepset/bert-base-cased-squad2", "deepset/bert-large-uncased-whole-word-masking-squad2",
                 "deepset/xlm-roberta-large-squad2", "distilbert-base-uncased-distilled-squad"]
reader_models_ci = ["deepset/minilm-uncased-squad2"]

reader_types = ["farm"]
data_dir = Path("../../data/squad20")
filename = "dev-v2.0.json"
# Note that this number is approximate - it was calculated using Bert Base Cased
# This number could vary when using a different tokenizer
n_total_passages = 12350
n_total_docs = 1204

results_file = "reader_results.csv"

reader_json_file = "../../docs/_src/benchmarks/reader_performance.json"

doc_index = "eval_document"
label_index = "label"

def benchmark_reader(ci=False, update_json=False, save_markdown=False, **kwargs):
    if ci:
        reader_models = reader_models_ci
    else:
        reader_models = reader_models_full
    reader_results = []
    doc_store = get_document_store("elasticsearch")
    # download squad data
    _download_extract_downstream_data(input_file=data_dir/filename)
    docs, labels = eval_data_from_json(data_dir/filename, max_docs=None)

    index_to_doc_store(doc_store, docs, None, labels)
    for reader_name in reader_models:
        for reader_type in reader_types:
            logger.info(f"##### Start reader run - model:{reader_name}, type: {reader_type} ##### ")
            try:
                reader = get_reader(reader_name, reader_type)
                results = reader.eval(document_store=doc_store,
                                      doc_index=doc_index,
                                      label_index=label_index,
                                      device="cuda")
                # print(results)
                results["passages_per_second"] = n_total_passages / results["reader_time"]
                results["reader"] = reader_name
                results["error"] = ""
                reader_results.append(results)
            except Exception as e:
                results = {'EM': 0.,
                           'f1': 0.,
                           'top_n_accuracy': 0.,
                           'top_n': 0,
                           'reader_time': 0.,
                           "passages_per_second": 0.,
                           "seconds_per_query": 0.,
                           'reader': reader_name,
                           "error": e}
                reader_results.append(results)
            reader_df = pd.DataFrame.from_records(reader_results)
            reader_df.to_csv(results_file)
            if save_markdown:
                md_file = results_file.replace(".csv", ".md")
                with open(md_file, "w") as f:
                    f.write(str(reader_df.to_markdown()))
    doc_store.delete_all_documents(label_index)
    doc_store.delete_all_documents(doc_index)
    if update_json:
        populate_reader_json()


def populate_reader_json():
    reader_results = reader_json()
    template = READER_TEMPLATE
    template["data"] = reader_results
    json.dump(template, open(reader_json_file, "w"), indent=4)


if __name__ == "__main__":
    benchmark_reader(ci=True, update_json=True, save_markdown=True)