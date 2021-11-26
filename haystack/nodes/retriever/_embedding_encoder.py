from typing import TYPE_CHECKING,  Callable, List, Union, Dict

if TYPE_CHECKING:
    from haystack.nodes.retriever import EmbeddingRetriever

import logging
from abc import abstractmethod
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.nn import DataParallel
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, AutoModel

from haystack.schema import Document
from haystack.modeling.data_handler.dataset import convert_features_to_dataset, flatten_rename
from haystack.modeling.utils import initialize_device_settings
from haystack.modeling.infer import Inferencer
from haystack.modeling.data_handler.dataloader import NamedDataLoader


logger = logging.getLogger(__name__)


class _BaseEmbeddingEncoder:
    @abstractmethod
    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries.

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        pass

    @abstractmethod
    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed
        :return: Embeddings, one per input document
        """
        pass


class _DefaultEmbeddingEncoder(_BaseEmbeddingEncoder):

    def __init__(
            self,
            retriever: 'EmbeddingRetriever'
    ):

        self.embedding_model = Inferencer.load(
            retriever.embedding_model, revision=retriever.model_version, task_type="embeddings",
            extraction_strategy=retriever.pooling_strategy,
            extraction_layer=retriever.emb_extraction_layer, gpu=retriever.use_gpu,
            batch_size=retriever.batch_size, max_seq_len=retriever.max_seq_len, num_processes=0,use_auth_token=retriever.use_auth_token
        )
        # Check that document_store has the right similarity function
        similarity = retriever.document_store.similarity
        # If we are using a sentence transformer model
        if "sentence" in retriever.embedding_model.lower() and similarity != "cosine":
            logger.warning(f"You seem to be using a Sentence Transformer with the {similarity} function. "
                           f"We recommend using cosine instead. "
                           f"This can be set when initializing the DocumentStore")
        elif "dpr" in retriever.embedding_model.lower() and similarity != "dot_product":
            logger.warning(f"You seem to be using a DPR model with the {similarity} function. "
                           f"We recommend using dot_product instead. "
                           f"This can be set when initializing the DocumentStore")

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[np.ndarray]:
        # TODO: FARM's `sample_to_features_text` need to fix following warning -
        # tokenization_utils.py:460: FutureWarning: `is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.
        emb = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])
        emb = [(r["vec"]) for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        passages = [d.content for d in docs] # type: ignore
        return self.embed(passages)


class _SentenceTransformersEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(
            self,
            retriever: 'EmbeddingRetriever'
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Can't find package `sentence-transformers` \n"
                              "You can install it via `pip install sentence-transformers` \n"
                              "For details see https://github.com/UKPLab/sentence-transformers ")
        # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
        # e.g. 'roberta-base-nli-stsb-mean-tokens'
        self.embedding_model = SentenceTransformer(retriever.embedding_model, device=str(retriever.devices[0]))
        self.batch_size = retriever.batch_size
        self.embedding_model.max_seq_length = retriever.max_seq_len
        self.show_progress_bar = retriever.progress_bar
        document_store = retriever.document_store
        if document_store.similarity != "cosine":
            logger.warning(
                f"You are using a Sentence Transformer with the {document_store.similarity} function. "
                f"We recommend using cosine instead. "
                f"This can be set when initializing the DocumentStore")

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[np.ndarray]:
        # texts can be a list of strings or a list of [title, text]
        # get back list of numpy embedding vectors
        emb = self.embedding_model.encode(texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar)
        emb = [r for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.content] for d in docs]  # type: ignore
        return self.embed(passages)


class _RetribertEmbeddingEncoder(_BaseEmbeddingEncoder):

    def __init__(
            self,
            retriever: 'EmbeddingRetriever'
    ):

        self.progress_bar = retriever.progress_bar
        self.batch_size = retriever.batch_size
        self.max_length = retriever.max_seq_len
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(retriever.embedding_model)
        self.embedding_model = AutoModel.from_pretrained(retriever.embedding_model).to(str(retriever.devices[0]))
      

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:

        queries = [{"text": q} for q in texts]
        dataloader = self._create_dataloader(queries)

        embeddings: List[np.ndarray] = []
        disable_tqdm = True if len(dataloader) == 1 else not self.progress_bar

        for i, batch in enumerate(tqdm(dataloader, desc=f"Creating Embeddings", unit=" Batches", disable=disable_tqdm)):
            batch = {key: batch[key].to(self.embedding_model.device) for key in batch}
            with torch.no_grad():
                q_reps = self.embedding_model.embed_questions(input_ids=batch["input_ids"],
                                                              attention_mask=batch["padding_mask"]).cpu().numpy()
            embeddings.append(q_reps)

        return np.concatenate(embeddings)

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:

        doc_text = [{"text": d.content} for d in docs]
        dataloader = self._create_dataloader(doc_text)

        embeddings: List[np.ndarray] = []
        disable_tqdm = True if len(dataloader) == 1 else not self.progress_bar

        for i, batch in enumerate(tqdm(dataloader, desc=f"Creating Embeddings", unit=" Batches", disable=disable_tqdm)):
            batch = {key: batch[key].to(self.embedding_model.device) for key in batch}
            with torch.no_grad():
                q_reps = self.embedding_model.embed_answers(input_ids=batch["input_ids"],
                                                            attention_mask=batch["padding_mask"]).cpu().numpy()
            embeddings.append(q_reps)

        return np.concatenate(embeddings)

    def _create_dataloader(self, text_to_encode: List[dict]) -> NamedDataLoader:

        dataset, tensor_names = self.dataset_from_dicts(text_to_encode)
        dataloader = NamedDataLoader(dataset=dataset, sampler=SequentialSampler(dataset),
                                     batch_size=self.batch_size, tensor_names=tensor_names)
        return dataloader

    def dataset_from_dicts(self, dicts: List[dict]):
        texts = [x["text"] for x in dicts]
        tokenized_batch = self.embedding_tokenizer(
            texts,
            return_token_type_ids=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
            padding=True
        )

        features_flat = flatten_rename(tokenized_batch,
                                       ["input_ids", "token_type_ids", "attention_mask"],
                                       ["input_ids", "segment_ids", "padding_mask"])
        dataset, tensornames = convert_features_to_dataset(features=features_flat)
        return dataset, tensornames


_EMBEDDING_ENCODERS: Dict[str, Callable] = {
    "farm": _DefaultEmbeddingEncoder,
    "transformers": _DefaultEmbeddingEncoder,
    "sentence_transformers": _SentenceTransformersEmbeddingEncoder,
    "retribert":_RetribertEmbeddingEncoder
}
