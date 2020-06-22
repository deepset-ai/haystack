import logging
from typing import Type, List, Union, Tuple
import torch
import numpy as np


from farm.infer import Inferencer

from haystack.database.base import Document, BaseDocumentStore
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.retriever.sparse import logger

from haystack.retriever.dpr_utils import HFBertEncoder, BertTensorizer, BertTokenizer,\
    Tensorizer, load_states_from_checkpoint

logger = logging.getLogger(__name__)


class DensePassageRetriever(BaseRetriever):
    """
        Retriever that uses a dual transformer encoder (one for query, one for passage). It's based on:
        Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." arXiv preprint arXiv:2004.04906 (2020).
        (See https://arxiv.org/abs/2004.04906).

        You can load the model checkpoints of the author's in the original format.
        See their readme for download instructions: https://github.com/facebookresearch/DPR#resources--data-formats

    """
    def __init__(self,
                 document_store: Type[BaseDocumentStore],
                 embedding_model: str,
                 gpu: bool = True,
                 batch_size=16,
                 do_lower_case=False,
                 encoder_model_type="hf_bert",
                 fp16=False,
                 fp16_opt_level='O1',
                 fix_ctx_encoder=False,
                 ):
        """

        :param document_store:
        :param embedding_model:
        :param gpu:
        :param batch_size:
        :param do_lower_case:
        :param encoder_model_type:
        :param fp16:
        :param fp16_opt_level:
        :param fix_ctx_encoder:
        """

        self.document_store = document_store
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.do_lower_case = do_lower_case
        self.encoder_model_type = encoder_model_type
        self.fix_ctx_encoder = fix_ctx_encoder

        # Load checkpoint (incl. additional model params)
        saved_state = load_states_from_checkpoint(self.embedding_model)
        self._set_encoder_params_from_state(saved_state.encoder_params)

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

    def retrieve(self, query: str, candidate_doc_ids: List[str] = None, top_k: int = 10) -> List[Document]:  # type: ignore
        query_emb = self.create_embedding_query(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb[0], top_k, candidate_doc_ids)
        return documents

    def create_embedding_query(self, texts: List[str]) -> List[Tuple[object, np.array]]:
        result = self._generate_batch_predictions(texts=texts, model=self.query_encoder,
                                                  tensorizer=self.tensorizer, batch_size=self.batch_size)
        return result

    def create_embedding_passage(self, texts: List[str]) -> List[Tuple[object, np.array]]:
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

            ctx_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            ctx_seg_batch = torch.zeros_like(ctx_ids_batch).cuda()
            ctx_attn_mask = tensorizer.get_attn_mask(ctx_ids_batch).cuda()
            with torch.no_grad():
                _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
            out = out.cpu()

            total += len(batch_token_tensors)

            results.extend([
                (out[i].view(-1).numpy())
                for i in range(out.size(0))
            ])

            if total % 10 == 0:
                logger.info('Encoded passages %d', total)

        return results

    def _prepare_model(self, encoder, saved_state, prefix):
        encoder.to(self.device)
        if self.fp16:
            try:
                import apex
                from apex import amp
                apex.amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            encoder, _ = amp.initialize(encoder, None, opt_level=self.fp16_opt_level)

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

    def _set_encoder_params_from_state(self, state):
        if state:
            params_to_save = ['do_lower_case', 'pretrained_model_cfg', 'encoder_model_type',
            'pretrained_file',
            'projection_dim', 'sequence_length']

            override_params = [(param, state[param]) for param in params_to_save if param in state]
            for param, value in override_params:
                if hasattr(self, param):
                    logger.warning('Overriding args parameter value from checkpoint state. Param = %s, value = %s',
                                   param,
                                   value)
                setattr(self, param, value)


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