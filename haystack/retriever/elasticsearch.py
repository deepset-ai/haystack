import logging
from typing import Type
from farm.infer import Inferencer

from haystack.database.base import Document, BaseDocumentStore
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store: Type[BaseDocumentStore], custom_query: str = None):
        """
        :param document_store: an instance of a DocumentStore to retrieve documents from.
        :param custom_query: query string as per Elasticsearch DSL with a mandatory question placeholder($question).

                             Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
                             that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
                             names must match with the filters dict supplied in self.retrieve().

                             An example custom_query:
                            {
                                "size": 10,
                                "query": {
                                    "bool": {
                                        "should": [{"multi_match": {
                                            "query": "${question}",                 // mandatory $question placeholder
                                            "type": "most_fields",
                                            "fields": ["text", "title"]}}],
                                        "filter": [                                 // optional custom filters
                                            {"terms": {"year": "${years}"}},
                                            {"terms": {"quarter": "${quarters}"}}],
                                    }
                                },
                            }

                             For this custom_query, a sample retrieve() could be:
                             self.retrieve(query="Why did the revenue increase?",
                                           filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
        """
        self.document_store = document_store
        self.custom_query = custom_query

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10) -> [Document]:
        documents = self.document_store.query(query, filters, top_k, self.custom_query)
        logger.info(f"Got {len(documents)} candidates from retriever")

        return documents


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: Type[BaseDocumentStore],
        embedding_model: str,
        gpu: bool = True,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
    ):
        """
        TODO
        :param document_store:
        :param embedding_model:
        :param gpu:
        :param model_format:
        """
        self.document_store = document_store
        self.model_format = model_format
        self.embedding_model = embedding_model
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer

        logger.info(f"Init retriever using embeddings of model {embedding_model}")
        if model_format == "farm" or model_format == "transformers":
            self.embedding_model = Inferencer.load(
                embedding_model, task_type="embeddings", gpu=gpu, batch_size=4, max_seq_len=512
            )

        elif model_format == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
            # e.g. 'roberta-base-nli-stsb-mean-tokens'
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise NotImplementedError

    def retrieve(self, query: str, candidate_doc_ids: [str] = None, top_k: int = 10) -> [Document]:
        query_emb = self.create_embedding(query)
        documents = self.document_store.query_by_embedding(query_emb, top_k, candidate_doc_ids)

        return documents

    def create_embedding(self, text):
        if self.model_format == "farm":
            res = self.embedding_model.extract_vectors(
                dicts=[{"text": text}],
                extraction_strategy=self.pooling_strategy,
                extraction_layer=self.emb_extraction_layer,
            )
            emb = list(res[0]["vec"])
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            res = self.embedding_model.encode([text])  # get back list of numpy embedding vectors
            emb = res[0].tolist()
        return emb
