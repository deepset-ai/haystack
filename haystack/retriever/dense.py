import logging
from typing import Type, List, Union, Tuple, Optional
import torch
import numpy as np
from pathlib import Path

from farm.infer import Inferencer

from haystack.database.base import Document, BaseDocumentStore
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.retriever.sparse import logger

from haystack.retriever.dpr_utils import DPRContextEncoder, DPRQuestionEncoder, DPRConfig, DPRContextEncoderTokenizer, \
    DPRQuestionEncoderTokenizer

logger = logging.getLogger(__name__)


class DensePassageRetriever(BaseRetriever):
    """
        Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
        See the original paper for more details:
        Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
        (https://arxiv.org/abs/2004.04906).
    """

    def __init__(self,
                 document_store: BaseDocumentStore,
                 question_embedding_model: str,
                 passage_embedding_model: str,
                 sequence_length: int = 256,
                 projection_dim: int = 0,
                 use_gpu: bool = True,
                 batch_size: int = 16,
                 do_lower_case: bool = False,
                 use_amp: str = None,
                 ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches the one of the original author's in the repository (https://github.com/facebookresearch/DPR)
        See their readme for manual download instructions: https://github.com/facebookresearch/DPR#resources--data-formats

        :Example:

            # remote model from FAIR
            >>> DensePassageRetriever(document_store=your_doc_store, embedding_model="dpr-bert-base-nq", use_gpu=True)
            # or from local path
            >>> DensePassageRetriever(document_store=your_doc_store, embedding_model="some_path/ber-base-encoder.cp", use_gpu=True)

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or remote name of model checkpoint. The format equals the 
                                one used by original author's in https://github.com/facebookresearch/DPR. 
                                Currently available remote names: "dpr-bert-base-nq" 
        :param use_gpu: Whether to use gpu or not
        :param batch_size: Number of questions or passages to encode at once
        :param do_lower_case: Whether to lower case the text input in the tokenizer
        :param encoder_model_type: 
        :param use_amp: Whether to use Automatix Mixed Precision optimization from apex's to improve speed and memory consumption.
        :param use_amp: Optional usage of Automatix Mixed Precision optimization from apex's to improve speed and memory consumption.
                        Choose `None` or AMP optimization level:
                              - None -> Not using amp at all
                              - 'O0' -> Regular FP32
                              - 'O1' -> Mixed Precision (recommended, if optimization wanted)
        """

        self.document_store = document_store
        self.question_embedding_model = question_embedding_model
        self.passage_embedding_model = passage_embedding_model
        self.batch_size = batch_size

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.use_amp = use_amp
        self.do_lower_case = do_lower_case
        self.sequence_length = sequence_length
        self.projection_dim = projection_dim

        # Init & Load Encoders
        self.question_config = DPRConfig(projection_dim=self.projection_dim, config=self.question_embedding_model, dropout=0.0)
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.question_embedding_model, config=self.question_config)
        self.query_encoder = DPRQuestionEncoder.from_pretrained(self.question_embedding_model, config=self.question_config)

        self.passage_config = DPRConfig(projection_dim=self.projection_dim, config=self.passage_embedding_model, dropout=0.0)
        self.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.passage_embedding_model, config=self.passage_config)
        self.passage_encoder = DPRContextEncoder.from_pretrained(self.passage_embedding_model, config=self.passage_config)

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        if index is None:
            index = self.document_store.index
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], top_k=top_k, filters=filters, index=index)
        return documents

    def embed_queries(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: queries to embed
        :return: embeddings, one per input queries
        """
        queries = [self._normalize_question(q) for q in texts]
        result = self._generate_batch_predictions(texts=queries, model=self.query_encoder,
                                                  tokenizer=self.query_tokenizer,
                                                  batch_size=self.batch_size)
        return result

    def embed_passages(self, texts: List[str], titles: Optional[List[str]] = None) -> List[np.array]:
        """
        Create embeddings for a list of passages using the passage encoder

        :param texts: passage to embed
        :param titles: passage title to also take into account during embedding
        :return: embeddings, one per input passage
        """
        result = self._generate_batch_predictions(texts=texts, titles=titles,
                                                  model=self.passage_encoder,
                                                  tokenizer=self.passage_tokenizer,
                                                  batch_size=self.batch_size)
        return result

    def _normalize_question(self, question: str) -> str:
        if question[-1] == '?':
            question = question[:-1]
        return question

    def _tensorizer(self, tokenizer, title: Optional[List[str]], text: List[str], add_special_tokens: bool = True):
        if title:
            texts = [tuple((title_, text_)) for title_, text_ in zip(title, text)]
            out = tokenizer.batch_encode_plus(texts, truncation=True,
                                              add_special_tokens=add_special_tokens,
                                              max_length=self.sequence_length,
                                              pad_to_max_length=True)
        else:
            out = tokenizer.batch_encode_plus(text, add_special_tokens=add_special_tokens, truncation=True,
                                              max_length=self.sequence_length,
                                              pad_to_max_length=True)

        token_ids = torch.tensor(out['input_ids'])
        token_type_ids = torch.tensor(out['token_type_ids'])
        attention_mask = torch.tensor(out['attention_mask'])
        return token_ids, token_type_ids, attention_mask

    def _generate_batch_predictions(self,
                                    texts: List[str],
                                    model: torch.nn.Module,
                                    tokenizer,
                                    titles: Optional[List[str]] = None, #useful only for passage embedding with DPR!
                                    batch_size: int = 16) -> List[Tuple[object, np.array]]:
        n = len(texts)
        total = 0
        results = []
        for batch_start in range(0, n, batch_size):
            ctx_title = titles[batch_start:batch_start + batch_size] if titles else None
            ctx_text = texts[batch_start:batch_start + batch_size]
            ctx_ids_batch, ctx_seg_batch, ctx_attn_mask = self._tensorizer(tokenizer, text=ctx_text, title=ctx_title)

            with torch.no_grad():
                out = model(input_ids=ctx_ids_batch, attention_mask=ctx_attn_mask, token_type_ids=ctx_seg_batch)
                # TODO revert back to when updating transformers
                # out = out.pooler_output
                out = out[0]
            out = out.cpu()

            total += ctx_ids_batch.size()[0]

            results.extend([
                (out[i].view(-1).numpy())
                for i in range(out.size(0))
            ])

            if total % 10 == 0:
                logger.info(f'Embedded {total} / {n} texts')

        return results

class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model: str,
        use_gpu: bool = True,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub. Example: 'deepset/sentence_bert'
        :param use_gpu: Whether to use gpu or not
        :param model_format: Name of framework that was used for saving the model. Options: 'farm', 'transformers', 'sentence_transformers'
        :param pooling_strategy: Strategy for combining the embeddings from the model (for farm / transformers models only).
                                 Options: 'cls_token' (sentence vector), 'reduce_mean' (sentence vector),
                                 reduce_max (sentence vector), 'per_token' (individual token vectors)
        :param emb_extraction_layer: Number of layer from which the embeddings shall be extracted (for farm / transformers models only).
                                     Default: -1 (very last layer).
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
                extraction_layer=self.emb_extraction_layer, gpu=use_gpu, batch_size=4, max_seq_len=512, num_processes=0
            )

        elif model_format == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
            # e.g. 'roberta-base-nli-stsb-mean-tokens'
            if use_gpu:
                device = "cuda"
            else:
                device = "cpu"
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        else:
            raise NotImplementedError

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        if index is None:
            index = self.document_store.index
        query_emb = self.embed(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], filters=filters,
                                                           top_k=top_k, index=index)
        return documents

    def embed(self, texts: Union[List[str], str]) -> List[np.array]:
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
            emb = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])  # type: ignore
            emb = [(r["vec"]) for r in emb]
        elif self.model_format == "sentence_transformers":
            # text is single string, sentence-transformers needs a list of strings
            # get back list of numpy embedding vectors
            emb = self.embedding_model.encode(texts)  # type: ignore
            # cast to float64 as float32 can cause trouble when serializing for ES
            emb = [(r.astype('float64')) for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of queries. For this Retriever type: The same as calling .embed()

        :param texts: queries to embed
        :return: embeddings, one per input queries
        """
        return self.embed(texts)

    def embed_passages(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of passages. For this Retriever type: The same as calling .embed()

        :param texts: passage to embed
        :return: embeddings, one per input passage
        """

        return self.embed(texts)
