# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import replace
from typing import Any

from more_itertools import batched
from openai import APIError, AsyncOpenAI, OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret
from haystack.utils.http_client import init_http_client

logger = logging.getLogger(__name__)


@component
class OpenAIDocumentEmbedder:
    """
    Computes document embeddings using OpenAI models.

    ### Usage example
    <!-- test-ignore -->
    ```python
    from haystack import Document
    from haystack.components.embedders import OpenAIDocumentEmbedder

    doc = Document(content="I love pizza!")
    document_embedder = OpenAIDocumentEmbedder()
    result = document_embedder.run([doc])

    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(  # noqa: PLR0913 (too-many-arguments)
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "text-embedding-ada-002",
        dimensions: int | None = None,
        api_base_url: str | None = None,
        organization: str | None = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        *,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Creates an OpenAIDocumentEmbedder component.

        Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the OpenAI client.

        :param api_key:
            The OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model:
            The name of the model to use for calculating embeddings.
            The default model is `text-embedding-ada-002`.
        :param dimensions:
            The number of dimensions of the resulting embeddings. Only `text-embedding-3` and
            later models support this parameter.
        :param api_base_url:
            Overrides the default base URL for all HTTP requests.
        :param organization:
            Your OpenAI organization ID. See OpenAI's
            [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
            for more information.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of documents to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar when running.
        :param meta_fields_to_embed:
            List of metadata fields to embed along with the document text.
        :param embedding_separator:
            Separator used to concatenate the metadata fields to the document text.
        :param timeout:
            Timeout for OpenAI client calls. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or 5 retries.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param raise_on_failure:
            Whether to raise an exception if the embedding request fails. If `False`, the component will log the error
            and continue processing the remaining documents. If `True`, it will raise an exception on failure.
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.api_base_url = api_base_url
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs
        self.raise_on_failure = raise_on_failure

        # Clients are built lazily at warm-up so the component can be constructed and deserialized without the
        # API key present; the async client is built on the serving loop. See warm_up / warm_up_async.
        self.client: OpenAI | None = None
        self.async_client: AsyncOpenAI | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        timeout = self.timeout if self.timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        max_retries = (
            self.max_retries if self.max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        )
        return {
            "api_key": self.api_key.resolve_value(),
            "organization": self.organization,
            "base_url": self.api_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }

    def warm_up(self) -> None:
        """
        Initializes the synchronous OpenAI client.
        """
        if self.client is None:
            self.client = OpenAI(
                http_client=init_http_client(self.http_client_kwargs, async_client=False), **self._client_kwargs()
            )

    async def warm_up_async(self) -> None:  # noqa: RUF029
        """
        Initializes the asynchronous OpenAI client on the serving event loop.
        """
        if self.async_client is None:
            self.async_client = AsyncOpenAI(
                http_client=init_http_client(self.http_client_kwargs, async_client=True), **self._client_kwargs()
            )

    def close(self) -> None:
        """
        Releases the synchronous OpenAI client.
        """
        if self.client is not None:
            self.client.close()
            self.client = None

    async def close_async(self) -> None:
        """
        Releases the asynchronous OpenAI client.
        """
        if self.async_client is not None:
            await self.async_client.close()
            self.async_client = None

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            model=self.model,
            dimensions=self.dimensions,
            api_base_url=self.api_base_url,
            organization=self.organization,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAIDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: list[Document]) -> dict[str, str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = {}
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            texts_to_embed[doc.id] = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )

        return texts_to_embed

    def _embed_batch(
        self, texts_to_embed: dict[str, str], batch_size: int
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        doc_ids_to_embeddings: dict[str, list[float]] = {}
        meta: dict[str, Any] = {}
        for batch in tqdm(
            batched(texts_to_embed.items(), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            args: dict[str, Any] = {"model": self.model, "input": [b[1] for b in batch], "encoding_format": "float"}

            if self.dimensions is not None:
                args["dimensions"] = self.dimensions

            try:
                # this method is invoked after warm_up, so client is not None
                assert self.client is not None
                response = self.client.embeddings.create(**args)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                msg = "Failed embedding of documents {ids} caused by {exc}"
                logger.exception(msg, ids=ids, exc=exc)
                if self.raise_on_failure:
                    raise exc
                continue

            embeddings = [el.embedding for el in response.data]
            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), embeddings, strict=True)))

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                meta["usage"]["total_tokens"] += response.usage.total_tokens

        return doc_ids_to_embeddings, meta

    async def _embed_batch_async(
        self, texts_to_embed: dict[str, str], batch_size: int
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        """
        Embed a list of texts in batches asynchronously.
        """

        doc_ids_to_embeddings: dict[str, list[float]] = {}
        meta: dict[str, Any] = {}

        batches = list(batched(texts_to_embed.items(), batch_size))
        if self.progress_bar:
            batches = async_tqdm(batches, desc="Calculating embeddings")

        for batch in batches:
            args: dict[str, Any] = {"model": self.model, "input": [b[1] for b in batch]}

            if self.dimensions is not None:
                args["dimensions"] = self.dimensions

            try:
                # this method is invoked after warm_up_async, so async_client is not None
                assert self.async_client is not None
                response = await self.async_client.embeddings.create(**args)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                msg = "Failed embedding of documents {ids} caused by {exc}"
                logger.exception(msg, ids=ids, exc=exc)
                if self.raise_on_failure:
                    raise exc
                continue

            embeddings = [el.embedding for el in response.data]
            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), embeddings, strict=True)))

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                meta["usage"]["total_tokens"] += response.usage.total_tokens

        return doc_ids_to_embeddings, meta

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embeds a list of documents.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "OpenAIDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OpenAITextEmbedder."
            )

        if self.client is None:
            self.warm_up()

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        doc_ids_to_embeddings, meta = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        new_documents = []
        for doc in documents:
            if doc.id in doc_ids_to_embeddings:
                new_documents.append(replace(doc, embedding=doc_ids_to_embeddings[doc.id]))
            else:
                new_documents.append(replace(doc))

        return {"documents": new_documents, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Embeds a list of documents asynchronously.

        :param documents:
            A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "OpenAIDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the OpenAITextEmbedder."
            )

        if self.async_client is None:
            await self.warm_up_async()

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        doc_ids_to_embeddings, meta = await self._embed_batch_async(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size
        )

        new_documents = []
        for doc in documents:
            if doc.id in doc_ids_to_embeddings:
                new_documents.append(replace(doc, embedding=doc_ids_to_embeddings[doc.id]))
            else:
                new_documents.append(replace(doc))

        return {"documents": new_documents, "meta": meta}
