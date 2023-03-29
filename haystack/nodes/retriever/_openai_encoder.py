import logging
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC
from haystack.nodes.retriever._base_embedding_encoder import _BaseEmbeddingEncoder
from haystack.schema import Document
from haystack.utils.openai_utils import USE_TIKTOKEN, count_openai_tokens, load_openai_tokenizer, openai_request
from haystack.telemetry import send_event

if TYPE_CHECKING:
    from haystack.nodes.retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))


class _OpenAIEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        send_event("OpenAIEmbeddingEncoder initialized", event_properties={"model": retriever.embedding_model})
        # See https://platform.openai.com/docs/guides/embeddings and
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/embeddings?tabs=console for more details
        self.using_azure = (
            retriever.azure_deployment_name is not None
            and retriever.azure_base_url is not None
            and retriever.api_version is not None
        )

        if self.using_azure:
            self.url = f"{retriever.azure_base_url}/openai/deployments/{retriever.azure_deployment_name}/embeddings?api-version={retriever.api_version}"
        else:
            self.url = "https://api.openai.com/v1/embeddings"

        self.api_key = retriever.api_key
        self.batch_size = min(64, retriever.batch_size)
        self.progress_bar = retriever.progress_bar
        model_class: str = next(
            (m for m in ["ada", "babbage", "davinci", "curie"] if m in retriever.embedding_model), "babbage"
        )

        tokenizer = self._setup_encoding_models(model_class, retriever.embedding_model, retriever.max_seq_len)
        self._tokenizer = load_openai_tokenizer(tokenizer_name=tokenizer)

    def _setup_encoding_models(self, model_class: str, model_name: str, max_seq_len: int):
        """
        Setup the encoding models for the retriever.
        """

        tokenizer_name = "gpt2"
        # new generation of embedding models (December 2022), we need to specify the full name
        if model_name.endswith("-002"):
            self.query_encoder_model = model_name
            self.doc_encoder_model = model_name
            self.max_seq_len = min(8191, max_seq_len)
            if USE_TIKTOKEN:
                from tiktoken.model import MODEL_TO_ENCODING

                tokenizer_name = MODEL_TO_ENCODING.get(model_name, "cl100k_base")
        else:
            self.query_encoder_model = f"text-search-{model_class}-query-001"
            self.doc_encoder_model = f"text-search-{model_class}-doc-001"
            self.max_seq_len = min(2046, max_seq_len)

        return tokenizer_name

    def _ensure_text_limit(self, text: str) -> str:
        """
        Ensure that length of the text is within the maximum length of the model.
        OpenAI v1 embedding models have a limit of 2046 tokens, and v2 models have a limit of 8191 tokens.
        """
        n_tokens = count_openai_tokens(text, self._tokenizer)
        if n_tokens <= self.max_seq_len:
            return text

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens to fit within the max token limit."
            " Reduce the length of the prompt to prevent it from being cut off.",
            n_tokens,
            self.max_seq_len,
        )

        if USE_TIKTOKEN:
            tokenized_payload = self._tokenizer.encode(text)
            decoded_string = self._tokenizer.decode(tokenized_payload[: self.max_seq_len])
        else:
            tokenized_payload = self._tokenizer.tokenize(text)
            decoded_string = self._tokenizer.convert_tokens_to_string(tokenized_payload[: self.max_seq_len])

        return decoded_string

    def embed(self, model: str, text: List[str]) -> np.ndarray:
        if self.api_key is None:
            raise ValueError(
                f"{'Azure ' if self.using_azure else ''}OpenAI API key is not set. You can set it via the `api_key` parameter of the EmbeddingRetriever."
            )

        generated_embeddings: List[Any] = []

        headers: Dict[str, str] = {"Content-Type": "application/json"}

        def azure_get_embedding(input: str):
            headers["api-key"] = str(self.api_key)
            azure_payload: Dict[str, str] = {"input": input}
            res = openai_request(url=self.url, headers=headers, payload=azure_payload, timeout=OPENAI_TIMEOUT)
            return res["data"][0]["embedding"]

        if self.using_azure:
            thread_count = cpu_count() if len(text) > cpu_count() else len(text)
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                results: Iterator[Dict[str, Any]] = executor.map(azure_get_embedding, text)
                generated_embeddings.extend(results)
        else:
            payload: Dict[str, Union[List[str], str]] = {"model": model, "input": text}
            headers["Authorization"] = f"Bearer {self.api_key}"

            res = openai_request(url=self.url, headers=headers, payload=payload, timeout=OPENAI_TIMEOUT)

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
