from haystack.retriever.base import BaseRetriever


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, datastore):
        self.datastore = datastore

    def retrieve(self, query, candidate_doc_ids=None, top_k=10):
        return self.datastore.query(query, top_k)

