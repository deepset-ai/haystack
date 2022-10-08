import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

import numpy as np
import requests
import torch
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from haystack.errors import OpenAIError
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.data_handler.dataset import convert_features_to_dataset, flatten_rename
from haystack.modeling.infer import Inferencer
from haystack.nodes.retriever._losses import _TRAINING_LOSSES
from haystack.schema import Document

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
        num_warmup_steps: int = None,
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
        # Check that document_store has the right similarity function
        similarity = retriever.document_store.similarity
        # If we are using a sentence transformer model
        if "sentence" in retriever.embedding_model.lower() and similarity != "cosine":
            logger.warning(
                f"You seem to be using a Sentence Transformer with the {similarity} function. "
                f"We recommend using cosine instead. "
                f"This can be set when initializing the DocumentStore"
            )
        elif "dpr" in retriever.embedding_model.lower() and similarity != "dot_product":
            logger.warning(
                f"You seem to be using a DPR model with the {similarity} function. "
                f"We recommend using dot_product instead. "
                f"This can be set when initializing the DocumentStore"
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
        num_warmup_steps: int = None,
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
        document_store = retriever.document_store
        if document_store.similarity != "cosine":
            logger.warning(
                f"You are using a Sentence Transformer with the {document_store.similarity} function. "
                f"We recommend using cosine instead. "
                f"This can be set when initializing the DocumentStore"
            )

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> np.ndarray:
        # texts can be a list of strings or a list of [title, text]
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
        passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.content] for d in docs]
        return self.embed(passages)

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: int = None,
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
            with torch.no_grad():
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
            with torch.no_grad():
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
        num_warmup_steps: int = None,
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
        # pretrained embedding models coming from:
        self.url = "https://api.openai.com/v1/embeddings"
        self.api_key = retriever.api_key
        model_class: str = next(
            (m for m in ["ada", "babbage", "davinci", "curie"] if m in retriever.embedding_model), "babbage"
        )
        self.query_model_encoder_engine = f"text-search-{model_class}-query-001"
        self.doc_model_encoder_engine = f"text-search-{model_class}-doc-001"

    def embed(self, model, texts: Union[List[str], str]) -> np.ndarray:
        payload = {"model": model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", self.url, headers=headers, data=json.dumps(payload))
        res = json.loads(response.text)

        if response.status_code != 200:
            raise OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}"
            )

        generated_embeddings = [ans["embedding"] for ans in res["data"]]
        return np.array(generated_embeddings)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.embed(self.query_model_encoder_engine, queries)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        return self.embed(self.doc_model_encoder_engine, [d.content for d in docs])

    def train(
        self,
        training_data: List[Dict[str, Any]],
        learning_rate: float = 2e-5,
        n_epochs: int = 1,
        num_warmup_steps: int = None,
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
}
