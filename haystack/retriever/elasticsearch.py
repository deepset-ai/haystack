from haystack.retriever.base import BaseRetriever
import logging
logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store):
        self.document_store = document_store

    def retrieve(self, query, candidate_doc_ids=None, top_k=10):
        paragraphs, meta_data = self.document_store.query(query, top_k, candidate_doc_ids)
        logger.info(f"Got {len(paragraphs)} candidates from retriever: {meta_data}")
        return paragraphs, meta_data
