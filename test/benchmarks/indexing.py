from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.converters import TextFileToDocument
from elasticsearch_haystack import ElasticsearchDocumentStore
from utils import get_docs
from time import perf_counter
import datetime

paths = get_docs("msmarco.1000")

document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")

pipe = Pipeline()
pipe.add_component(instance=TextFileToDocument(), name="converter")
pipe.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
pipe.connect("converter", "writer")


def benchmark_indexing():
    start_time = perf_counter()
    pipe.run({"converter": {"sources": paths}})
    end_time = perf_counter()

    indexing_time = end_time - start_time
    n_docs = len(paths)

    doc_store_type = type(document_store).__name__

    results = {
        "doc_store": doc_store_type,
        "n_docs": n_docs,
        "indexing_time": indexing_time,
        "docs_per_second": n_docs / indexing_time,
        "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": None,
    }

    return results
