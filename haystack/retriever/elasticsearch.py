import logging

from farm.infer import Inferencer

from haystack.database.base import Document
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store, embedding_model=None, gpu=True, model_format="farm",
                 pooling_strategy="reduce_mean", emb_extraction_layer=-1, direct_filters=None,
                 custom_query=None):
        """
        TODO
        :param document_store:
        :param embedding_model:
        :param gpu:
        :param model_format:
        """
        self.document_store = document_store
        self.model_format = model_format
        self.embedding_model = None
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.direct_filters = direct_filters
        self.custom_query = custom_query

        # only needed if you want to retrieve via cosinge similarity of embeddings
        if embedding_model:
            logger.info(f"Init retriever using embeddings of model {embedding_model}")
            if model_format == "farm" or model_format == "transformers":
                self.embedding_model = Inferencer.load(embedding_model, task_type="embeddings",
                                                   gpu=gpu, batch_size=4, max_seq_len=512)

            elif model_format == "sentence_transformers":
                from sentence_transformers import SentenceTransformer
                # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
                # e.g. 'roberta-base-nli-stsb-mean-tokens'
                self.embedding_model = SentenceTransformer(embedding_model)
            else:
                raise NotImplementedError

    def retrieve(self, query: str, candidate_doc_ids: [str] = None, top_k: int = 10) -> [Document]:
        if self.embedding_model:
            # cos. similarity of embeddings
            query_emb = self.create_embedding(query)
            documents = self.document_store.query_by_embedding(query_emb, top_k, candidate_doc_ids)
        else:
            # regular ES query (e.g. BM25)
            documents = self.document_store.query(query, top_k, candidate_doc_ids)
        logger.info(f"Got {len(documents)} candidates from retriever")
        return documents

    def create_embedding(self, text):
        if self.model_format == "farm":
            res = self.embedding_model.extract_vectors(dicts=[{"text": text}],
                                                       extraction_strategy=self.pooling_strategy,
                                                       extraction_layer=self.emb_extraction_layer)
            emb = list(res[0]["vec"])
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            res = self.embedding_model.encode([text]) # get back list of numpy embedding vectors
            emb = res[0].tolist()
        return emb
