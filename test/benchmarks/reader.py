from utils import get_document_store, index_to_doc_store, get_reader
from pathlib import Path
import pandas as pd

reader_models = ["deepset/roberta-base-squad2", "deepset/minilm-uncased-squad2", "deepset/bert-base-cased-squad2", "deepset/bert-large-uncased-whole-word-masking-squad2", "deepset/xlm-roberta-large-squad2"]
reader_types = ["farm"]
data_dir = Path("../../data/squad20")
filename = "dev-v2.0.json"
n_passages = 12247

doc_index = "eval_document"
label_index = "label"

def benchmark():
    reader_results = []
    doc_store = get_document_store("elasticsearch")
    index_to_doc_store(doc_store, data_dir, filename, None)
    for reader_name in reader_models:
        for reader_type in reader_types:
            try:
                reader = get_reader(reader_name, reader_type)
                results = reader.eval(document_store=doc_store,
                                      doc_index=doc_index,
                                      label_index=label_index,
                                      device="cuda")
                # print(results)
                results["passages_per_second"] = n_passages / results["reader_time"]
                results["reader"] = reader_name
                results["error"] = ""
                reader_results.append(results)
            except Exception as e:
                results = {'EM': 0., 'f1': 0., 'top_n_accuracy': 0., 'reader_time': 0., "passages_per_second": 0, 'reader': reader_name, "error": e}
                reader_results.append(results)
            reader_df = pd.DataFrame.from_records(reader_results)
            reader_df.to_csv("reader_results.csv")


if __name__ == "__main__":
    benchmark()