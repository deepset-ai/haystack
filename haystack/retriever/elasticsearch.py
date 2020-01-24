from haystack.retriever.base import BaseRetriever


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store):
        self.document_store = document_store

    def retrieve(self, query, candidate_doc_ids=None, top_k=10):
        return self.document_store.query(query, top_k)
