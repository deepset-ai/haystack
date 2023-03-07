import json
from pathlib import Path

from canals.pipeline import Pipeline
from canals.stores import MemoryDocumentStore
from canals.data import TextQuery, TextDocument
from canals.nodes import RetrieveByBM25, ReadByTransformers

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    document_store = MemoryDocumentStore()
    document_store.write_documents(
        [
            TextDocument(content="My name is Anna and I live in Paris."),
            TextDocument(content="My name is Serena and I live in Rome."),
            TextDocument(content="My name is Julia and I live in Berlin."),
        ]
    )
    pipeline = Pipeline()
    pipeline.add_store("my_documents", document_store)
    pipeline.add_node("retriever", RetrieveByBM25(default_store="my_documents"))
    pipeline.add_node(
        "reader",
        ReadByTransformers(model_name_or_path="distilbert-base-uncased-distilled-squad"),
    )

    pipeline.connect(["retriever", "reader"])
    pipeline.draw(tmp_path / "query_pipeline.png")

    results = pipeline.run({"query": TextQuery(content="Who lives in Berlin?")})

    results["answers_by_query"] = {str(key): value for key, value in results["answers_by_query"].items()}
    print(json.dumps(results, indent=4, default=repr))


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
