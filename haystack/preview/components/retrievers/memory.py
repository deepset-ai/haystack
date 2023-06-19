from dataclasses import dataclass
from typing import Dict, List, Any

from haystack.preview import component, Document, ComponentInput, ComponentOutput
from haystack.preview.document_stores import MemoryDocumentStore


@component
class MemoryRetriever:
    @dataclass
    class Input(ComponentInput):
        query: str
        top_k: int
        scale_score: bool
        stores: Dict[str, Any]

    @dataclass
    class Output(ComponentOutput):
        documents: List[Document]

    def __init__(self, document_store_name: str, top_k: int = 10, scale_score: bool = True):
        self.document_store_name = document_store_name
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.defaults = {"top_k": top_k, "scale_score": scale_score}

    def run(self, data: Input) -> Output:
        if self.document_store_name not in data.stores:
            raise ValueError(
                f"MemoryRetriever's document store '{self.document_store_name}' not found "
                f"in input stores {list(data.stores.keys())}"
            )
        document_store = data.stores[self.document_store_name]
        if not isinstance(document_store, MemoryDocumentStore):
            raise ValueError("MemoryRetriever can only be used with a MemoryDocumentStore instance.")
        docs = document_store.bm25_retrieval(query=data.query, top_k=data.top_k, scale_score=data.scale_score)
        return MemoryRetriever.Output(documents=docs)
