import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import numpy as np
import requests
import torch
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from haystack.environment import (
    HAYSTACK_REMOTE_API_BACKOFF_SEC,
    HAYSTACK_REMOTE_API_MAX_RETRIES,
    HAYSTACK_REMOTE_API_TIMEOUT_SEC,
)
from haystack.errors import CohereError
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.data_handler.dataset import convert_features_to_dataset, flatten_rename
from haystack.modeling.infer import Inferencer
from haystack.nodes.retriever._losses import _TRAINING_LOSSES
from haystack.nodes.retriever._openai_encoder import _OpenAIEmbeddingEncoder
from haystack.schema import Document
from haystack.utils.reflection import retry_with_exponential_backoff

from ._base_embedding_encoder import _BaseEmbeddingEncoder

if TYPE_CHECKING:
    from haystack.nodes.retriever import EmbeddingRetriever


COHERE_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
COHERE_BACKOFF = float(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 10))
COHERE_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


logger = logging.getLogger(__name__)


class _DefaultEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        self.embedding_model = Inferencer.load(
            retriever.embedding_model,
            revision=retriever.model_version,
            task_type="embeddings",
            extraction_strategy=retriever.pooling_strategy,
            extraction_layer=retriever.emb_extraction_layer,
            gpu=retriever.use_gpu,
            batch_size=retriever.batch_size,
            max_seq_len=retriever.max_seq_len,
            num_processes=0,
            use_auth_token=retriever.use_auth_token,
        )
        if retriever.document_store:
            self._check_docstore_similarity_function(
                document_store=retriever.document_store, model_name=retriever.embedding_model
            )

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> np.ndarray:
        # TODO: FARM's `sample_to_features_text` need to fix following warning -
        # tokenization_utils.py:460: FutureWarning: `is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.
        emb = self.embedding_model.inference_from_dicts(dicts=[{"text": t} for t in texts])
        emb = np.stack([r["vec"] for r in emb])
        return emb

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        return self.embed(queries)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        passages = [d.content for d in docs]
        return self.embed(passages)

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: Optional[int] = None,
        batch_size: int = 16,
        train_loss: Literal["mnrl", "margin_mse"] = "mnrl",
        num_workers: int = 0,
        use_amp: bool = False,
        **kwargs,
    ):
        raise NotImplementedError(
            "You can't train this retriever. You can only use the `train` method with sentence-transformers EmbeddingRetrievers."
        )

    def save(self, save_dir: Union[Path, str]):
        raise NotImplementedError(
            "You can't save your record as `save` only works for sentence-transformers EmbeddingRetrievers."
        )


class _SentenceTransformersEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
        # e.g. 'roberta-base-nli-stsb-mean-tokens'
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, ModuleNotFoundError) as ie:
            from haystack.utils.import_utils import _optional_component_not_installed

            _optional_component_not_installed(__name__, "sentence", ie)

        self.embedding_model = SentenceTransformer(
            retriever.embedding_model, device=str(retriever.devices[0]), use_auth_token=retriever.use_auth_token
        )
        self.batch_size = retriever.batch_size
        self.embedding_model.max_seq_length = retriever.max_seq_len
        self.show_progress_bar = retriever.progress_bar
        if retriever.document_store:
            self._check_docstore_similarity_function(
                document_store=retriever.document_store, model_name=retriever.embedding_model
            )

    def embed(self, texts: Union[List[str], str]) -> np.ndarray:
        # texts can be a list of strings
        # get back list of numpy embedding vectors
        emb = self.embedding_model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        return emb

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        return self.embed(queries)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        passages = [d.content for d in docs]
        return self.embed(passages)

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: Optional[int] = None,
        batch_size: Optional[int] = 16,
        train_loss: Literal["mnrl", "margin_mse"] = "mnrl",
        num_workers: int = 0,
        use_amp: bool = False,
        **kwargs,
    ):
        """
        Trains the underlying Sentence Transformer model.

        Each training data example is a dictionary with the following keys:

        * question: The question string.
        * pos_doc: Positive document string (the document containing the answer).
        * neg_doc: Negative document string (the document that doesn't contain the answer).
        * score: The score margin the answer must fall within.

        :param training_data: The training data in a dictionary format.
        :param learning_rate: The learning rate of the optimizer.
        :param n_epochs: The number of iterations on the whole training data set you want to train for.
        :param num_warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is
            increased from 0 up to the maximal learning rate. After these many training steps, the learning rate is
            decreased linearly back to zero.
        :param batch_size: The batch size to use for the training. The default value is 16.
        :param train_loss: Specify the training loss to use to fit the Sentence-Transformers model. Possible options are
            "mnrl" (Multiple Negatives Ranking Loss) and "margin_mse".
        :param num_workers: The number of subprocesses to use for the Pytorch DataLoader.
        :param use_amp: Use Automatic Mixed Precision (AMP).
        :param kwargs: Additional training keyword arguments to pass to the `SentenceTransformer.fit` function. Please
            reference the Sentence-Transformers [documentation](https://www.sbert.net/docs/training/overview.html#sentence_transformers.SentenceTransformer.fit)
            for a full list of keyword arguments.
        """

        if train_loss not in _TRAINING_LOSSES:
            raise ValueError(f"Unrecognized train_loss {train_loss}. Should be one of: {_TRAINING_LOSSES.keys()}")

        st_loss = _TRAINING_LOSSES[train_loss]

        train_examples = []
        for train_i in training_data:
            missing_attrs = st_loss.required_attrs.difference(set(train_i.keys()))
            if len(missing_attrs) > 0:
                raise ValueError(
                    f"Some training examples don't contain the fields {missing_attrs} which are necessary when using the '{train_loss}' loss."
                )

            texts = [train_i["question"], train_i["pos_doc"]]
            if "neg_doc" in train_i:
                texts.append(train_i["neg_doc"])

            if "score" in train_i:
                train_examples.append(InputExample(texts=texts, label=train_i["score"]))
            else:
                train_examples.append(InputExample(texts=texts))

        logger.info("Training/adapting %s with %s examples", self.embedding_model, len(train_examples))
        train_dataloader = DataLoader(
            train_examples,  # type: ignore [var-annotated, arg-type]
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
        )
        train_loss = st_loss.loss(self.embedding_model)

        # Tune the model
        self.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=n_epochs,
            optimizer_params={"lr": learning_rate},
            warmup_steps=int(len(train_dataloader) * 0.1) if num_warmup_steps is None else num_warmup_steps,
            use_amp=use_amp,
            **kwargs,
        )

    def save(self, save_dir: Union[Path, str]):
        self.embedding_model.save(path=str(save_dir))


class _RetribertEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        self.progress_bar = retriever.progress_bar
        self.batch_size = retriever.batch_size
        self.max_length = retriever.max_seq_len
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            retriever.embedding_model, use_auth_token=retriever.use_auth_token
        )
        self.embedding_model = AutoModel.from_pretrained(
            retriever.embedding_model, use_auth_token=retriever.use_auth_token
        ).to(str(retriever.devices[0]))

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        query_text = [{"text": q} for q in queries]
        dataloader = self._create_dataloader(query_text)

        embeddings: List[np.ndarray] = []
        disable_tqdm = True if len(dataloader) == 1 else not self.progress_bar

        for batch in tqdm(dataloader, desc="Creating Embeddings", unit=" Batches", disable=disable_tqdm):
            batch = {key: batch[key].to(self.embedding_model.device) for key in batch}
            with torch.inference_mode():
                q_reps = (
                    self.embedding_model.embed_questions(
                        input_ids=batch["input_ids"], attention_mask=batch["padding_mask"]
                    )
                    .cpu()
                    .numpy()
                )
            embeddings.append(q_reps)

        return np.concatenate(embeddings)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        doc_text = [{"text": d.content} for d in docs]
        dataloader = self._create_dataloader(doc_text)

        embeddings: List[np.ndarray] = []
        disable_tqdm = True if len(dataloader) == 1 else not self.progress_bar

        for batch in tqdm(dataloader, desc="Creating Embeddings", unit=" Batches", disable=disable_tqdm):
            batch = {key: batch[key].to(self.embedding_model.device) for key in batch}
            with torch.inference_mode():
                q_reps = (
                    self.embedding_model.embed_answers(
                        input_ids=batch["input_ids"], attention_mask=batch["padding_mask"]
                    )
                    .cpu()
                    .numpy()
                )
            embeddings.append(q_reps)

        return np.concatenate(embeddings)

    def _create_dataloader(self, text_to_encode: List[dict]) -> NamedDataLoader:
        dataset, tensor_names = self.dataset_from_dicts(text_to_encode)
        dataloader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        return dataloader

    def dataset_from_dicts(self, dicts: List[dict]):
        texts = [x["text"] for x in dicts]
        tokenized_batch = self.embedding_tokenizer(
            texts,
            return_token_type_ids=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )

        features_flat = flatten_rename(
            tokenized_batch,
            ["input_ids", "token_type_ids", "attention_mask"],
            ["input_ids", "segment_ids", "padding_mask"],
        )
        dataset, tensornames = convert_features_to_dataset(features=features_flat)
        return dataset, tensornames

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: Optional[int] = None,
        batch_size: int = 16,
        train_loss: Literal["mnrl", "margin_mse"] = "mnrl",
        num_workers: int = 0,
        use_amp: bool = False,
        **kwargs,
    ):
        raise NotImplementedError(
            "You can't train this retriever. You can only use the `train` method with sentence-transformers EmbeddingRetrievers."
        )

    def save(self, save_dir: Union[Path, str]):
        raise NotImplementedError(
            "You can't save your record as `save` only works for sentence-transformers EmbeddingRetrievers."
        )


class _CohereEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # See https://docs.cohere.ai/embed-reference/ for more details
        # Cohere has a max seq length of 4096 tokens and a max batch size of 96
        self.max_seq_len = min(4096, retriever.max_seq_len)
        self.url = "https://api.cohere.ai/embed"
        self.api_key = retriever.api_key
        self.batch_size = min(96, retriever.batch_size)
        self.progress_bar = retriever.progress_bar
        self.model: str = next(
            (
                m
                for m in ["small", "medium", "large", "multilingual-22-12", "finance-sentiment"]
                if m in retriever.embedding_model
            ),
            "multilingual-22-12",
        )

    @retry_with_exponential_backoff(
        backoff_in_seconds=COHERE_BACKOFF, max_retries=COHERE_MAX_RETRIES, errors=(CohereError,)
    )
    def embed(self, model: str, text: List[str]) -> np.ndarray:
        payload = {"model": model, "texts": text, "truncate": "END"}
        headers = {"Authorization": f"BEARER {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", self.url, headers=headers, data=json.dumps(payload), timeout=COHERE_TIMEOUT)
        res = json.loads(response.text)
        if response.status_code != 200:
            raise CohereError(response.text, status_code=response.status_code)
        generated_embeddings = [e for e in res["embeddings"]]
        return np.array(generated_embeddings)

    def embed_batch(self, text: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(
            range(0, len(text), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = text[i : i + self.batch_size]
            generated_embeddings = self.embed(self.model, batch)
            all_embeddings.append(generated_embeddings)
        return np.concatenate(all_embeddings)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.embed_batch(queries)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        return self.embed_batch([d.content for d in docs])

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: Optional[int] = None,
        batch_size: int = 16,
        train_loss: Literal["mnrl", "margin_mse"] = "mnrl",
        num_workers: int = 0,
        use_amp: bool = False,
        **kwargs,
    ):
        raise NotImplementedError(f"Training is not implemented for {self.__class__}")

    def save(self, save_dir: Union[Path, str]):
        raise NotImplementedError(f"Saving is not implemented for {self.__class__}")


_EMBEDDING_ENCODERS: Dict[str, Callable] = {
    "farm": _DefaultEmbeddingEncoder,
    "transformers": _DefaultEmbeddingEncoder,
    "sentence_transformers": _SentenceTransformersEmbeddingEncoder,
    "retribert": _RetribertEmbeddingEncoder,
    "openai": _OpenAIEmbeddingEncoder,
    "cohere": _CohereEmbeddingEncoder,
}
