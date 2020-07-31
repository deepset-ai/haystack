import logging
from typing import Type, List, Union, Tuple
import torch
import numpy as np
from pathlib import Path

from farm.infer import Inferencer

from haystack.database.base import Document, BaseDocumentStore
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.retriever.sparse import logger

from haystack.retriever.dpr_utils import HFBertEncoder, BertTensorizer, BertTokenizer,\
    Tensorizer, load_states_from_checkpoint, download_dpr

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
                 embedding_model: str,
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
        self.embedding_model = embedding_model
        self.batch_size = batch_size

        #TODO Proper Download + Caching of model if not locally available
        if embedding_model == "dpr-bert-base-nq":
            if not Path("models/dpr/checkpoint/retriever/single/nq/bert-base-encoder.cp").is_file():
                download_dpr(resource_key="checkpoint.retriever.single.nq.bert-base-encoder", out_dir="models/dpr")
            self.embedding_model = "models/dpr/checkpoint/retriever/single/nq/bert-base-encoder.cp"

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.use_amp = use_amp
        self.do_lower_case = do_lower_case

        # Load checkpoint (incl. additional model params)
        saved_state = load_states_from_checkpoint(self.embedding_model)
        logger.info('Loaded encoder params:  %s', saved_state.encoder_params)
        self.do_lower_case = saved_state.encoder_params["do_lower_case"]
        self.pretrained_model_cfg = saved_state.encoder_params["pretrained_model_cfg"]
        self.encoder_model_type = saved_state.encoder_params["encoder_model_type"]
        self.pretrained_file = saved_state.encoder_params["pretrained_file"]
        self.projection_dim = saved_state.encoder_params["projection_dim"]
        self.sequence_length = saved_state.encoder_params["sequence_length"]

        # Init & Load Encoders
        self.query_encoder = HFBertEncoder.init_encoder(self.pretrained_model_cfg,
                                                        projection_dim=self.projection_dim,
                                                        dropout=0.0)
        self.passage_encoder = HFBertEncoder.init_encoder(self.pretrained_model_cfg,
                                                          projection_dim=self.projection_dim,
                                                          dropout=0.0)
        self.passage_encoder = self._prepare_model(self.passage_encoder, saved_state, prefix="ctx_model.")
        self.query_encoder = self._prepare_model(self.query_encoder, saved_state, prefix="question_model.")
        #self.encoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=self.fix_ctx_encoder)

        # Load Tokenizer & Tensorizer
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_cfg, do_lower_case=self.do_lower_case)
        self.tensorizer = BertTensorizer(tokenizer, self.sequence_length)

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
        result = self._generate_batch_predictions(texts=texts, model=self.query_encoder,
                                                  tensorizer=self.tensorizer, batch_size=self.batch_size)
        return result

    def embed_passages(self, texts: List[str]) -> List[np.array]:
        """
        Create embeddings for a list of passages using the passage encoder

        :param texts: passage to embed
        :return: embeddings, one per input passage
        """
        result = self._generate_batch_predictions(texts=texts, model=self.passage_encoder,
                                                  tensorizer=self.tensorizer, batch_size=self.batch_size)
        return result

    def _generate_batch_predictions(self,
                                    texts: List[str],
                                    model: torch.nn.Module,
                                    tensorizer: Tensorizer,
                                    batch_size: int = 16) -> List[Tuple[object, np.array]]:
        n = len(texts)
        total = 0
        results = []
        for j, batch_start in enumerate(range(0, n, batch_size)):

            batch_token_tensors = [tensorizer.text_to_tensor(ctx) for ctx in
                                   texts[batch_start:batch_start + batch_size]]

            ctx_ids_batch = torch.stack(batch_token_tensors, dim=0).to(self.device)
            ctx_seg_batch = torch.zeros_like(ctx_ids_batch).to(self.device)
            ctx_attn_mask = tensorizer.get_attn_mask(ctx_ids_batch).to(self.device)
            with torch.no_grad():
                _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
            out = out.cpu()

            total += len(batch_token_tensors)

            results.extend([
                (out[i].view(-1).numpy())
                for i in range(out.size(0))
            ])

            if total % 10 == 0:
                logger.info(f'Embedded {total} / {n} texts')

        return results

    def _prepare_model(self, encoder, saved_state, prefix):
        encoder.to(self.device)
        if self.use_amp:
            try:
                import apex
                from apex import amp
                apex.amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            encoder, _ = amp.initialize(encoder, None, opt_level=self.use_amp)

        encoder.eval()

        # load weights from the model file
        model_to_load = encoder.module if hasattr(encoder, 'module') else encoder
        logger.info('Loading saved model state ...')
        logger.debug('saved model keys =%s', saved_state.model_dict.keys())

        prefix_len = len(prefix)
        ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                     key.startswith(prefix)}
        model_to_load.load_state_dict(ctx_state)
        return encoder


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
