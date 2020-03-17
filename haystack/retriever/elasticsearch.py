from haystack.retriever.base import BaseRetriever
from farm.infer import Inferencer

import logging
logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store):
        self.document_store = document_store

    def retrieve(self, query, candidate_doc_ids=None, top_k=10):
        paragraphs, meta_data = self.document_store.query(query, top_k, candidate_doc_ids)
        logger.info(f"Got {len(paragraphs)} candidates from retriever: {meta_data}")
        return paragraphs, meta_data


class ElasticsearchEmbeddingRetriever(BaseRetriever):
    def __init__(self, document_store, embedding_model, gpu):
        self.document_store = document_store
        self.embedding_model = Inferencer.load(embedding_model, task_type="embeddings",
                                               gpu=gpu, batch_size=4, max_seq_len=512)

    def retrieve(self, query, candidate_doc_ids=None, top_k=10):
        query_emb = self.create_embedding(query)
        paragraphs, meta_data = self.document_store.query_by_embedding(query_emb, top_k, candidate_doc_ids)
        logger.info(f"Got {len(paragraphs)} candidates from retriever: {meta_data}")
        return paragraphs, meta_data

    def create_embedding(self, text):
        return self.embedding_model.extract_vectors(text, extraction_strategy="reduce_mean", extraction_layer=-1)