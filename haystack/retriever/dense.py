import logging
from typing import Type, List, Union
from farm.infer import Inferencer

from haystack.database.base import Document, BaseDocumentStore
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.retriever.sparse import logger

logger = logging.getLogger(__name__)


class DPRRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: Type[BaseDocumentStore],
        embedding_model: str,
        gpu: bool = True,
    ):
        """
        TODO
        :param document_store:
        :param embedding_model:
        :param gpu:
        :param model_format:
        """
        self.document_store = document_store
        self.embedding_model = embedding_model


        logger.info(f"Init retriever using embeddings of model {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)


    def retrieve(self, query: str, candidate_doc_ids: [str] = None, top_k: int = 10) -> [Document]:
        query_emb = self.create_embedding(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb[0], top_k, candidate_doc_ids)

        return documents

    def create_embedding(self, texts: [str]):
        """
        Create embeddings for each text in a list of texts using the retrievers model (`self.embedding_model`)
        :param texts: texts to embed
        :return: list of embeddings (one per input text). Each embedding is a list of floats.
        """

        # for backward compatibility: cast pure str input
        if type(texts) == str:
            texts = [texts]
        assert type(texts) == list, "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"

        if self.model_format == "farm":
            res = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])
            emb = [list(r["vec"]) for r in res] #cast from numpy
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            res = self.embedding_model.encode(texts)  # get back list of numpy embedding vectors
            emb = [list(r) for r in res] #cast from numpy
        return emb


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: ElasticsearchDocumentStore,
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
                embedding_model, task_type="embeddings", extraction_strategy=self.pooling_strategy,
                extraction_layer=self.emb_extraction_layer, gpu=gpu, batch_size=4, max_seq_len=512, num_processes=0
            )

        elif model_format == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
            # e.g. 'roberta-base-nli-stsb-mean-tokens'
            if gpu:
                device = "cuda"
            else:
                device = "cpu"
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        else:
            raise NotImplementedError

    def retrieve(self, query: str, candidate_doc_ids: List[str] = None, top_k: int = 10) -> List[Document]:  # type: ignore
        query_emb = self.create_embedding(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb[0], top_k, candidate_doc_ids)
        return documents

    def create_embedding(self, texts: Union[List[str], str]) -> List[List[float]]:
        """
        Create embeddings for each text in a list of texts using the retrievers model (`self.embedding_model`)
        :param texts: texts to embed
        :return: list of embeddings (one per input text). Each embedding is a list of floats.
        """

        # for backward compatibility: cast pure str input
        if type(texts) == str:
            texts = [texts]  # type: ignore
        assert type(texts) == list, "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"

        if self.model_format == "farm" or self.model_format == "transformers":
            res = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])  # type: ignore
            emb = [list(r["vec"]) for r in res] #cast from numpy
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            # get back list of numpy embedding vectors
            res = self.embedding_model.encode(texts)  # type: ignore
            emb = [list(r.astype('float64')) for r in res] #cast from numpy
        return emb