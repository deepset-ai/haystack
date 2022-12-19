import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any, Callable, Dict, List, Union

import numpy as np
import requests
import torch
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from haystack.document_stores.base import BaseDocumentStore

from haystack.errors import OpenAIError, OpenAIRateLimitError, CohereError
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.data_handler.dataset import convert_features_to_dataset, flatten_rename
from haystack.modeling.infer import Inferencer
from haystack.nodes.retriever._losses import _TRAINING_LOSSES
from haystack.schema import Document
from haystack.utils.reflection import retry_with_exponential_backoff

if TYPE_CHECKING:
    from haystack.nodes.retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)


class _BaseEmbeddingEncoder:
    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        pass

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: Optional[int] = None,
        batch_size: int = 16,
    ):
        """
        Trains or adapts the underlying embedding model.

        Each training data example is a dictionary with the following keys:

        * question: The question string.
        * pos_doc: Positive document string (the document containing the answer).
        * neg_doc: Negative document string (the document that doesn't contain the answer).
        * score: The score margin the answer must fall within.


        :param training_data: The training data in a dictionary format. Required.
        :type training_data: List[Dict[str, Any]]
        :param learning_rate: The speed at which the model learns. Required. We recommend that you leave the default `2e-5` value.
        :type learning_rate: float
        :param n_epochs: The number of epochs (complete passes of the training data through the algorithm) that you want the model to go through. Required.
        :type n_epochs: int
        :param num_warmup_steps: The number of warmup steps for the model. Warmup steps are epochs when the learning rate is very low. You can use them at the beginning of the training to prevent early overfitting of your model. Required.
        :type num_warmup_steps: int
        :param batch_size: The batch size to use for the training. Optional. The default values is 16.
        :type batch_size: int (optional)
        """
        pass

    def save(self, save_dir: Union[Path, str]):
        """
        Save the model to the directory you specify.

        :param save_dir: The directory where the model is saved. Required.
        :type save_dir: Union[Path, str]
        """
        pass

    def _check_docstore_similarity_function(self, document_store: BaseDocumentStore, model_name: str):
        """
        Check that document_store uses a similarity function
        compatible with the embedding model
        """
        if "sentence-transformers" in model_name.lower():
            model_similarity = None
            if "-cos-" in model_name.lower():
                model_similarity = "cosine"
            elif "-dot-" in model_name.lower():
                model_similarity = "dot_product"

            if model_similarity is not None and document_store.similarity != model_similarity:
                logger.warning(
                    f"You seem to be using {model_name} model with the {document_store.similarity} function instead of the recommended {model_similarity}. "
                    f"This can be set when initializing the DocumentStore"
                )
        elif "dpr" in model_name.lower() and document_store.similarity != "dot_product":
            logger.warning(
                f"You seem to be using a DPR model with the {document_store.similarity} function. "
                f"We recommend using dot_product instead. "
                f"This can be set when initializing the DocumentStore"
            )


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
        batch_size: int = 16,
        train_loss: str = "mnrl",
    ):

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
        train_dataloader = DataLoader(train_examples, batch_size=batch_size, drop_last=True, shuffle=True)
        train_loss = st_loss.loss(self.embedding_model)

        # Tune the model
        self.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=n_epochs,
            optimizer_params={"lr": learning_rate},
            warmup_steps=int(len(train_dataloader) * 0.1) if num_warmup_steps is None else num_warmup_steps,
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

        for i, batch in enumerate(tqdm(dataloader, desc=f"Creating Embeddings", unit=" Batches", disable=disable_tqdm)):
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

        for i, batch in enumerate(tqdm(dataloader, desc=f"Creating Embeddings", unit=" Batches", disable=disable_tqdm)):
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
    ):
        raise NotImplementedError(
            "You can't train this retriever. You can only use the `train` method with sentence-transformers EmbeddingRetrievers."
        )

    def save(self, save_dir: Union[Path, str]):
        raise NotImplementedError(
            "You can't save your record as `save` only works for sentence-transformers EmbeddingRetrievers."
        )


class _OpenAIEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # See https://beta.openai.com/docs/guides/embeddings for more details
        # OpenAI has a max seq length of 2048 tokens and unknown max batch size
        self.max_seq_len = min(2048, retriever.max_seq_len)
        self.url = "https://api.openai.com/v1/embeddings"
        self.api_key = retriever.api_key
        self.batch_size = min(64, retriever.batch_size)
        self.progress_bar = retriever.progress_bar
        model_class: str = next(
            (m for m in ["ada", "babbage", "davinci", "curie"] if m in retriever.embedding_model), "babbage"
        )
        self._setup_encoding_models(model_class, retriever.embedding_model)

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def _setup_encoding_models(self, model_class: str, model_name: str):
        """
        Setup the encoding models for the retriever.
        """
        # new generation of embedding models (December 2022), we need to specify the full name
        if "text-embedding" in model_name:
            self.query_encoder_model = model_name
            self.doc_encoder_model = model_name
        else:
            self.query_encoder_model = f"text-search-{model_class}-query-001"
            self.doc_encoder_model = f"text-search-{model_class}-doc-001"

    def _ensure_text_limit(self, text: str) -> str:
        """
        Ensure that length of the text is within the maximum length of the model.
        OpenAI embedding models have a limit of 2048 tokens
        """
        tokenized_payload = self.tokenizer(text)
        return self.tokenizer.decode(tokenized_payload["input_ids"][: self.max_seq_len])

    @retry_with_exponential_backoff(backoff_in_seconds=10, max_retries=5)
    def embed(self, model: str, text: List[str]) -> np.ndarray:
        payload = {"model": model, "input": text}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", self.url, headers=headers, data=json.dumps(payload), timeout=30)
        res = json.loads(response.text)

        if response.status_code != 200:
            openai_error: OpenAIError
            if response.status_code == 429:
                openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
            else:
                openai_error = OpenAIError(
                    f"OpenAI returned an error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response body: {response.text}",
                    status_code=response.status_code,
                )
            raise openai_error

        unordered_embeddings = [(ans["index"], ans["embedding"]) for ans in res["data"]]
        ordered_embeddings = sorted(unordered_embeddings, key=lambda x: x[0])
        generated_embeddings = [emb[1] for emb in ordered_embeddings]
        return np.array(generated_embeddings)

    def embed_batch(self, model: str, text: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(
            range(0, len(text), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = text[i : i + self.batch_size]
            batch_limited = [self._ensure_text_limit(content) for content in batch]
            generated_embeddings = self.embed(model, batch_limited)
            all_embeddings.append(generated_embeddings)
        return np.concatenate(all_embeddings)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.embed_batch(self.query_encoder_model, queries)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        return self.embed_batch(self.doc_encoder_model, [d.content for d in docs])

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: Optional[int] = None,
        batch_size: int = 16,
    ):
        raise NotImplementedError(f"Training is not implemented for {self.__class__}")

    def save(self, save_dir: Union[Path, str]):
        raise NotImplementedError(f"Saving is not implemented for {self.__class__}")


class _CohereEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # See https://docs.cohere.ai/embed-reference/ for more details
        # Cohere has a max seq length of 4096 tokens and a max batch size of 16
        self.max_seq_len = min(4096, retriever.max_seq_len)
        self.url = "https://api.cohere.ai/embed"
        self.api_key = retriever.api_key
        self.batch_size = min(16, retriever.batch_size)
        self.progress_bar = retriever.progress_bar
        self.model: str = next(
            (
                m
                for m in ["small", "medium", "large", "multilingual-22-12", "finance-sentiment"]
                if m in retriever.embedding_model
            ),
            "multilingual-22-12",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def _ensure_text_limit(self, text: str) -> str:
        """
        Ensure that length of the text is within the maximum length of the model.
        Cohere embedding models have a limit of 4096 tokens
        """
        tokenized_payload = self.tokenizer(text)
        return self.tokenizer.decode(tokenized_payload["input_ids"][: self.max_seq_len])

    @retry_with_exponential_backoff(backoff_in_seconds=10, max_retries=5, errors=(CohereError,))
    def embed(self, model: str, text: List[str]) -> np.ndarray:
        payload = {"model": model, "texts": text}
        headers = {"Authorization": f"BEARER {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", self.url, headers=headers, data=json.dumps(payload), timeout=30)
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
            batch_limited = [self._ensure_text_limit(content) for content in batch]
            generated_embeddings = self.embed(self.model, batch_limited)
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
