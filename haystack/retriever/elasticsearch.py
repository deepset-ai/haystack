from haystack.retriever.base import BaseRetriever
from farm.infer import Inferencer
from sentence_transformers import SentenceTransformer


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
    def __init__(self, document_store, embedding_model, gpu, model_type="farm"):
        self.document_store = document_store
        self.model_type = model_type

        if model_type == "farm" or model_type == "transformers":
            self.embedding_model = Inferencer.load(embedding_model, task_type="embeddings",
                                               gpu=gpu, batch_size=4, max_seq_len=512)
        elif model_type == "sentence_transformers":
            # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
            # e.g. 'roberta-base-nli-stsb-mean-tokens'
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise NotImplementedError

    def retrieve(self, query, candidate_doc_ids=None, top_k=10):
        query_emb = self.create_embedding(query)
        paragraphs, meta_data = self.document_store.query_by_embedding(query_emb, top_k, candidate_doc_ids)
        logger.info(f"Got {len(paragraphs)} candidates from retriever: {meta_data}")
        return paragraphs, meta_data

    def create_embedding(self, text):
        if self.model_type == "farm":
            res = self.embedding_model.extract_vectors(dicts=[{"text": text}], extraction_strategy="reduce_mean", extraction_layer=-1)
            emb = list(res[0]["vec"])
        elif self.model_type == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            res = self.embedding_model.encode([text]) # get back list of numpy embedding vectors
            emb = res[0].tolist()
        return emb