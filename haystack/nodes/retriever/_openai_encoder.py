import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import requests
from tqdm.auto import tqdm

from haystack.environment import (
    HAYSTACK_REMOTE_API_BACKOFF_SEC,
    HAYSTACK_REMOTE_API_MAX_RETRIES,
    HAYSTACK_REMOTE_API_TIMEOUT_SEC,
)
from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.nodes.retriever._base_embedding_encoder import _BaseEmbeddingEncoder
from haystack.schema import Document
from haystack.utils.reflection import retry_with_exponential_backoff

if TYPE_CHECKING:
    from haystack.nodes.retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)

machine = platform.machine().lower()
system = platform.system()

USE_TIKTOKEN = False
if sys.version_info >= (3, 8) and (machine in ["amd64", "x86_64"] or (machine == "arm64" and system == "Darwin")):
    USE_TIKTOKEN = True

if USE_TIKTOKEN:
    import tiktoken  # pylint: disable=import-error
else:
    logger.warning(
        "OpenAI tiktoken module is not available for Python < 3.8,Linux ARM64 and AARCH64. Falling back to GPT2TokenizerFast."
    )
    from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast


OPENAI_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
OPENAI_BACKOFF = float(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 10))
OPENAI_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


logger = logging.getLogger(__name__)


class _OpenAIEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "EmbeddingRetriever"):
        # See https://beta.openai.com/docs/guides/embeddings for more details
        self.url = "https://api.openai.com/v1/embeddings"
        self.api_key = retriever.api_key
        self.batch_size = min(64, retriever.batch_size)
        self.progress_bar = retriever.progress_bar
        model_class: str = next(
            (m for m in ["ada", "babbage", "davinci", "curie"] if m in retriever.embedding_model), "babbage"
        )

        tokenizer = self._setup_encoding_models(model_class, retriever.embedding_model, retriever.max_seq_len)

        if USE_TIKTOKEN:
            logger.debug("Using tiktoken %s tokenizer", tokenizer)
            self._tk_tokenizer: tiktoken.Encoding = tiktoken.get_encoding(tokenizer)
        else:
            logger.debug("Using GPT2TokenizerFast tokenizer")
            self._hf_tokenizer: PreTrainedTokenizerFast = GPT2TokenizerFast.from_pretrained(tokenizer)

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
                tokenizer_name = "cl100k_base"
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

        if USE_TIKTOKEN:
            tokenized_payload = self._tk_tokenizer.encode(text)
            decoded_string = self._tk_tokenizer.decode(tokenized_payload[: self.max_seq_len])
        else:
            tokenized_payload = self._hf_tokenizer.tokenize(text)
            decoded_string = self._hf_tokenizer.convert_tokens_to_string(tokenized_payload[: self.max_seq_len])

        return decoded_string

    @retry_with_exponential_backoff(
        backoff_in_seconds=OPENAI_BACKOFF, max_retries=OPENAI_MAX_RETRIES, errors=(OpenAIRateLimitError, OpenAIError)
    )
    def embed(self, model: str, text: List[str]) -> np.ndarray:
        payload = {"model": model, "input": text}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.request("POST", self.url, headers=headers, data=json.dumps(payload), timeout=OPENAI_TIMEOUT)
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
