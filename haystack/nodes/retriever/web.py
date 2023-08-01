import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from typing import Dict, Iterator, List, Optional, Literal, Union
from unicodedata import combining, normalize

from haystack import Document
from haystack.document_stores.base import BaseDocumentStore
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.link_content import LinkContentFetcher
from haystack.nodes.search_engine.web import SearchEngine
from haystack.nodes.search_engine.web import WebSearch
from haystack.schema import FilterType

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    url: str
    snippet: str
    score: float
    position: Optional[str]


class WebRetriever(BaseRetriever):
    """
    The WebRetriever is an effective tool designed to extract relevant documents from the web. It leverages the WebSearch
    class to obtain web page results, strips the HTML from those pages, and extracts the raw text content. Depending on
    the operation mode, this text can be further broken down into smaller documents with the help of a PreProcessor.

    The WebRetriever supports three distinct modes of operation:

    - Snippets Mode: In this mode, the WebRetriever generates a list of Document instances, where each Document
    represents a snippet or a segment from a web page result. It's important to note that this mode does not involve
    actual web page retrieval.

    - Raw Documents Mode: In this mode, the WebRetriever generates a list of Document instances, where each Document
    represents an entire web page (retrieved from the search result link) devoid of any HTML and containing only the raw
    text content.

    - Preprocessed Documents Mode: This mode is similar to the Raw Documents Mode but includes an additional step -
    the raw text from each retrieved web page is divided into smaller Document instances using a specified PreProcessor.
    If no PreProcessor is specified, the default PreProcessor is used.
    """

    def __init__(
        self,
        api_key: str,
        search_engine_provider: Union[str, SearchEngine] = "SerperDev",
        top_search_results: Optional[int] = 10,
        top_k: Optional[int] = 5,
        mode: Literal["snippets", "raw_documents", "preprocessed_documents"] = "snippets",
        preprocessor: Optional[PreProcessor] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: int = 1 * 24 * 60 * 60,
    ):
        """
        :param api_key: API key for the search engine provider.
        :param search_engine_provider: Name of the search engine provider class, see `providers.py` for a list of supported providers.
        :param top_search_results: Number of top search results to be retrieved.
        :param top_k: Top k documents to be returned by the retriever.
        :param mode: Whether to return snippets, raw documents, or preprocessed documents. Snippets are the default.
        :param preprocessor: Optional PreProcessor to be used to split documents into paragraphs. If not provided, the default PreProcessor is used.
        :param cache_document_store: DocumentStore to be used to cache search results.
        :param cache_index: Index name to be used to cache search results.
        :param cache_headers: Headers to be used to cache search results.
        :param cache_time: Time in seconds to cache search results. Defaults to 24 hours.
        """
        super().__init__()
        self.web_search = WebSearch(
            api_key=api_key, top_k=top_search_results, search_engine_provider=search_engine_provider
        )
        self.mode = mode
        self.cache_document_store = cache_document_store
        self.document_store = cache_document_store
        self.cache_index = cache_index
        self.cache_headers = cache_headers
        self.cache_time = cache_time
        self.top_k = top_k
        self.preprocessor = None
        if preprocessor is not None:
            self.preprocessor = preprocessor
        elif mode == "preprocessed_documents":
            self.preprocessor = PreProcessor(progress_bar=False)

    def _normalize_query(self, query: str) -> str:
        return "".join([c for c in normalize("NFKD", query.lower()) if not combining(c)])

    def _check_cache(
        self,
        query: str,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: Optional[int] = None,
    ) -> List[Document]:
        """
        Private method to check if the documents for a given query are already cached. The documents are fetched from
        the specified DocumentStore. It retrieves documents that are newer than the cache_time limit.

        :param query: The query string to check in the cache.
        :param cache_index: Optional index name in the DocumentStore to fetch the documents. Defaults to the instance's
        cache_index.
        :param cache_headers: Optional headers to be used when fetching documents from the DocumentStore. Defaults to
        the instance's cache_headers.
        :param cache_time: Optional time limit in seconds to check the cache. Only documents newer than cache_time are
        returned. Defaults to the instance's cache_time.
        :returns: A list of Document instances fetched from the cache. If no documents are found in the cache, an empty
        list is returned.
        """
        cache_document_store = self.cache_document_store
        documents = []

        if cache_document_store is not None:
            query_norm = self._normalize_query(query)
            cache_filter: FilterType = {"$and": {"search.query": query_norm}}

            if cache_time is not None and cache_time > 0:
                cache_filter["timestamp"] = {
                    "$gt": int((datetime.utcnow() - timedelta(seconds=cache_time)).timestamp())
                }
                logger.debug("Cache filter: %s", cache_filter)

            documents = cache_document_store.get_all_documents(
                filters=cache_filter, index=cache_index, headers=cache_headers, return_embedding=False
            )

        logger.debug("Found %d documents in cache", len(documents))

        return documents

    def _save_cache(
        self,
        query: str,
        documents: List[Document],
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: Optional[int] = None,
    ) -> bool:
        """
        Private method to cache the retrieved documents for a given query.
        The documents are saved in the specified DocumentStore. If the same document already exists, it is
        overwritten.

        :param query: The query string for which the documents are being cached.
        :param documents: The list of Document instances to be cached.
        :param cache_index: Optional index name in the DocumentStore to save the documents. Defaults to the
        instance's cache_index.
        :param cache_headers: Optional headers to be used when saving documents in the DocumentStore. Defaults to
        the instance's cache_headers.
        :param cache_time: Optional time limit in seconds to check the cache. Documents older than the
        cache_time are deleted. Defaults to the instance's cache_time.
        :returns: True if the documents are successfully saved in the cache, False otherwise.
        """
        cache_document_store = self.cache_document_store

        if cache_document_store is not None:
            cache_document_store.write_documents(
                documents=documents, index=cache_index, headers=cache_headers, duplicate_documents="overwrite"
            )

            logger.debug("Saved %d documents in the cache", len(documents))

            cache_filter: FilterType = {"$and": {"search.query": query}}

            if cache_time is not None and cache_time > 0:
                cache_filter["timestamp"] = {
                    "$lt": int((datetime.utcnow() - timedelta(seconds=cache_time)).timestamp())
                }

                cache_document_store.delete_documents(index=cache_index, headers=cache_headers, filters=cache_filter)

                logger.debug("Deleted documents in the cache using filter: %s", cache_filter)

            return True

        return False

    def retrieve(  # type: ignore[override]
        self,
        query: str,
        top_k: Optional[int] = None,
        preprocessor: Optional[PreProcessor] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: Optional[int] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieve Documents in real-time from the web based on the URLs provided by the WebSearch.

        This method takes a search query as input, retrieves the corresponding web documents, and
        returns them in a structured format suitable for further processing or analysis. The documents
        are retrieved at runtime, ensuring up-to-date information.

        Optionally, the retrieved documents can be stored in a DocumentStore for future use, saving time
        and resources on repeated retrievals. This caching mechanism can significantly improve retrieval times
        for frequently accessed information.

        :param query: The query string.
        :param top_k: The number of Documents to be returned by the retriever.
        :param preprocessor: The PreProcessor to be used to split documents into paragraphs.
        :param cache_document_store: The DocumentStore to cache the documents to.
        :param cache_index: The index name to save the documents to.
        :param cache_headers: The headers to save the documents to.
        :param cache_time: The time limit in seconds to check the cache. The default is 24 hours.
        """

        # Initialize default parameters
        preprocessor = preprocessor or self.preprocessor
        cache_document_store = cache_document_store or self.cache_document_store
        cache_index = cache_index or self.cache_index
        cache_headers = cache_headers or self.cache_headers
        cache_time = cache_time or self.cache_time
        top_k = top_k or self.top_k

        # Normalize query
        query_norm = self._normalize_query(query)

        # Check cache for query
        extracted_docs = self._check_cache(
            query_norm, cache_index=cache_index, cache_headers=cache_headers, cache_time=cache_time
        )

        # If query is not cached, fetch from web
        if not extracted_docs:
            extracted_docs = self._retrieve_from_web(query_norm, preprocessor)

        # Save results to cache
        if cache_document_store and extracted_docs:
            cached = self._save_cache(query_norm, extracted_docs, cache_index=cache_index, cache_headers=cache_headers)
            if not cached:
                logger.warning(
                    "Could not save retrieved Documents to the DocumentStore cache. "
                    "Check your DocumentStore configuration."
                )
        return extracted_docs[:top_k]

    def _retrieve_from_web(self, query_norm: str, preprocessor: Optional[PreProcessor]) -> List[Document]:
        """
        Retrieve Documents from the web based on the query.

        :param query_norm: The normalized query string.
        :param preprocessor: The PreProcessor to be used to split documents into paragraphs.
        :return: List of Document objects.
        """

        search_results, _ = self.web_search.run(query=query_norm)
        search_results_docs = search_results["documents"]
        if self.mode == "snippets":
            return search_results_docs
        else:
            links: List[SearchResult] = self._prepare_links(search_results_docs)
            logger.debug("Starting to fetch %d links from WebSearch results", len(links))
            return self._scrape_links(links, query_norm, preprocessor)

    def _prepare_links(self, search_results: List[Document]) -> List[SearchResult]:
        """
        Prepare a list of SearchResult objects based on the search results from the search engine.

        :param search_results: List of Document objects obtained from web search.
        :return: List of SearchResult objects.
        """
        if not search_results:
            return []

        links: List[SearchResult] = [
            SearchResult(r.meta["link"], r.content, float(r.meta.get("score", 0.0)), r.meta.get("position"))
            for r in search_results
            if r.meta.get("link")
        ]
        return links

    def _scrape_links(
        self, links: List[SearchResult], query_norm: str, preprocessor: Optional[PreProcessor]
    ) -> List[Document]:
        """
        Scrape the links and return the documents.

        :param links: List of SearchResult objects.
        :param query_norm: The normalized query string.
        :param preprocessor: The PreProcessor object to be used to split documents into paragraphs.
        :return: List of Document objects obtained from scraping the links.
        """
        if not links:
            return []

        fetcher = (
            LinkContentFetcher(processor=preprocessor, raise_on_failure=True)
            if self.mode == "preprocessed_documents" and preprocessor
            else LinkContentFetcher(raise_on_failure=True)
        )

        def scrape_link_content(link: SearchResult) -> List[Document]:
            """
            Encapsulate the link scraping logic in a function to be used in a ThreadPoolExecutor.
            """
            docs: List[Document] = []
            try:
                docs = fetcher.fetch(
                    url=link.url,
                    doc_kwargs={
                        "search.score": link.score,
                        "id_hash_keys": ["meta.url", "meta.search.query"],
                        "search.query": query_norm,
                        "search.position": link.position,
                        "snippet_text": link.snippet,
                    },
                )
            except Exception as e:
                # Log the exception for debugging
                logger.debug("Error fetching documents from %s : %s", link.url, str(e))

            return docs

        thread_count = cpu_count() if len(links) > cpu_count() else len(links)
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            scraped_pages: Iterator[List[Document]] = executor.map(scrape_link_content, links)

        # Flatten list of lists to a single list
        extracted_docs = [doc for doc_list in scraped_pages for doc in doc_list]

        # Sort by score
        extracted_docs = sorted(extracted_docs, key=lambda x: x.meta["search.score"], reverse=True)

        return extracted_docs

    def retrieve_batch(  # type: ignore[override]
        self,
        queries: List[str],
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        preprocessor: Optional[PreProcessor] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: Optional[int] = None,
    ) -> List[List[Document]]:
        """
        Batch retrieval method that fetches documents for a list of queries. Each query is passed to the `retrieve`
        method which fetches documents from the web in real-time or from a DocumentStore cache. The fetched documents
        are then extended to a list of documents.

        :param queries: List of query strings to retrieve documents for.
        :param top_p: The number of documents to be returned by the retriever for each query. If None, the instance's
        default value is used.
        :param top_k: The maximum number of documents to be retrieved for each query. If None, the instance's default
        value is used.
        :param preprocessor: The PreProcessor to be used to split documents into paragraphs. If None, the instance's
        default PreProcessor is used.
        :param cache_document_store: The DocumentStore to cache the documents to. If None, the instance's default
        DocumentStore is used.
        :param cache_index: The index name to save the documents to. If None, the instance's default cache_index is used.
        :param cache_headers: The headers to save the documents to. If None, the instance's default cache_headers is used.
        :param cache_time: The time limit in seconds to check the cache. If None, the instance's default cache_time is used.
        :returns: A list of lists where each inner list represents the documents fetched for a particular query.
        """
        documents = []
        for q in queries:
            documents.append(
                self.retrieve(
                    query=q,
                    top_k=top_k,
                    preprocessor=preprocessor,
                    cache_document_store=cache_document_store,
                    cache_index=cache_index,
                    cache_headers=cache_headers,
                    cache_time=cache_time,
                )
            )

        return documents
