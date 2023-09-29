import json
import os
from typing import Optional, Dict, Union, List, Any
import logging

import requests

from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC, HAYSTACK_REMOTE_API_MAX_RETRIES
from haystack.errors import CohereInferenceLimitError, CohereUnauthorizedError, CohereError
from haystack.utils import request_with_retry

from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker


logger = logging.getLogger(__name__)


TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


class CohereRanker(BaseRanker):
    """
    You can use re-ranking on top of a Retriever to boost the performance for document search.
    This is particularly useful if the Retriever has a high recall but is bad in sorting the documents by relevance.

    Cohere models are trained with a context length of 512 tokens - the model takes into account both the input
    from the query and document. If your query is larger than 256 tokens, it will be truncated to the first 256 tokens.

    Cohere breaks down a query-document pair into 512 token chunks. For example, if your query is 50 tokens and your
    document is 1024 tokens, your document will be broken into the following chunks:
    ```bash
    relevance_score_1 = <query[0,50], document[0,460]>
    relevance_score_2 = <query[0,50], document[460,920]>
    relevance_score_3 = <query[0,50], document[920,1024]>
    relevance_score = max(relevance_score_1, relevance_score_2, relevance_score_3)
    ```

    Find more best practices for reranking in the [Cohere documentation](https://docs.cohere.com/docs/reranking-best-practices).
    """

    def __init__(
        self,
        api_key: str,
        model_name_or_path: str,
        top_k: int = 10,
        max_chunks_per_doc: Optional[int] = None,
        embed_meta_fields: Optional[List[str]] = None,
    ):
        """
         Creates an instance of CohereInvocationLayer for the specified Cohere model.

        :param api_key: Cohere API key.
        :param model_name_or_path: Cohere model name. Check the list of supported models in the [Cohere documentation](https://docs.cohere.com/docs/models).
        :param top_k: The maximum number of documents to return.
        :param max_chunks_per_doc: If your document exceeds 512 tokens, this determines the maximum number of
            chunks a document can be split into. If None, the default of 10 is used.
            For example, if your document is 6000 tokens, with the default of 10, the document will be split into 10
            chunks each of 512 tokens and the last 880 tokens will be disregarded.
        :param embed_meta_fields: Concatenate the provided meta fields and into the text passage that is then used in
            reranking. The original documents are returned so the concatenated metadata is not included in the returned documents.
        """
        super().__init__()
        valid_api_key = isinstance(api_key, str) and api_key
        if not valid_api_key:
            raise ValueError(
                f"api_key {api_key} must be a valid Cohere token. "
                f"Your token is available in your Cohere settings page."
            )
        # See model info at https://docs.cohere.com/docs/models
        # supported models are rerank-english-v2.0, rerank-multilingual-v2.0
        valid_model_name_or_path = isinstance(model_name_or_path, str) and model_name_or_path
        if not valid_model_name_or_path:
            raise ValueError(f"model_name_or_path {model_name_or_path} must be a valid Cohere model name")
        self.model_name_or_path = model_name_or_path
        self.api_key = api_key
        self.top_k = top_k
        self.max_chunks_per_doc = max_chunks_per_doc
        self.embed_meta_fields = embed_meta_fields

    @property
    def url(self) -> str:
        return "https://api.cohere.ai/v1/rerank"

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Request-Source": "python-sdk",
        }

    def _post(
        self,
        data: Dict[str, Any],
        attempts: int = RETRIES,
        status_codes_to_retry: Optional[List[int]] = None,
        timeout: float = TIMEOUT,
    ) -> requests.Response:
        """
        Post data to the Cohere re-ranker model. It takes in a query and a list of documents and returns a response
        using a REST invocation.

        :param data: The data to be sent to the model.
        :param attempts: The number of attempts to make.
        :param status_codes_to_retry: The status codes to retry on.
        :param timeout: The timeout for the request.
        :return: The response from the model as a requests.Response object.
        """
        response: requests.Response
        if status_codes_to_retry is None:
            status_codes_to_retry = [429]
        try:
            response = request_with_retry(
                method="POST",
                status_codes_to_retry=status_codes_to_retry,
                attempts=attempts,
                url=self.url,
                headers=self.headers,
                json=data,
                timeout=timeout,
            )
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:
                raise CohereInferenceLimitError(f"API rate limit exceeded: {res.text}")
            if res.status_code == 401:
                raise CohereUnauthorizedError(f"API key is invalid: {res.text}")

            raise CohereError(
                f"Cohere model returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",
                status_code=res.status_code,
            )
        return response

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Use the Cohere Reranker to re-rank the supplied list of documents based on the query.

        :param query: The query string.
        :param documents: List of Document to be re-ranked.
        :param top_k: The maximum number of documents to return.
        """
        if top_k is None:
            top_k = self.top_k

        # See https://docs.cohere.com/reference/rerank-1
        docs_with_meta_fields = self._add_meta_fields_to_docs(
            documents=documents, embed_meta_fields=self.embed_meta_fields
        )
        cohere_docs = [{"text": d.content} for d in docs_with_meta_fields]
        if len(cohere_docs) > 1000:
            logger.warning(
                "The Cohere reranking endpoint only supports 1000 documents. "
                "The number of documents has been truncated to 1000 from %s.",
                len(cohere_docs),
            )
            cohere_docs = cohere_docs[:1000]

        params = {
            "model": self.model_name_or_path,
            "query": query,
            "documents": cohere_docs,
            "top_n": None,  # By passing None we return all documents and use top_k to truncate later
            "return_documents": False,
            "max_chunks_per_doc": self.max_chunks_per_doc,
        }
        response = self._post(params)
        output = json.loads(response.text)

        indices = [o["index"] for o in output["results"]]
        scores = [o["relevance_score"] for o in output["results"]]
        sorted_docs = []
        for idx, score in zip(indices, scores):
            doc = documents[idx]
            doc.score = score
            sorted_docs.append(documents[idx])

        return sorted_docs[:top_k]

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        """
        Use Cohere Reranking endpoint to re-rank the supplied lists of Documents.

        Returns a lists of Documents sorted by (descending) similarity with the corresponding queries.

        - If you provide a list containing a single query...

            - ... and a single list of Documents, the single list of Documents will be re-ranked based on the
              supplied query.
            - ... and a list of lists of Documents, each list of Documents will be re-ranked individually based on the
              supplied query.

        - If you provide a list of multiple queries...

            - ... you need to provide a list of lists of Documents. Each list of Documents will be re-ranked based on
              its corresponding query.

        :param queries: List of queries.
        :param documents: Single list of Documents or list of lists of Documents to be reranked.
        :param top_k: The maximum number of documents to return per Document list.
        :param batch_size: Not relevant.
        """
        if top_k is None:
            top_k = self.top_k

        if len(documents) > 0 and isinstance(documents[0], Document):
            # Docs case 1: single list of Documents -> rerank single list of Documents based on single query
            if len(queries) != 1:
                raise HaystackError("Number of queries must be 1 if a single list of Documents is provided.")
            return self.predict(query=queries[0], documents=documents, top_k=top_k)  # type: ignore
        else:
            # Docs case 2: list of lists of Documents -> rerank each list of Documents based on corresponding query
            # If queries contains a single query, apply it to each list of Documents
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")

            results = []
            for query, cur_docs in zip(queries, documents):
                assert isinstance(cur_docs, list)
                results.append(self.predict(query=query, documents=cur_docs, top_k=top_k))  # type: ignore
            return results
