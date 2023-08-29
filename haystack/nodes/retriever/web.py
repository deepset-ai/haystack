import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from typing import Dict, Iterator, List, Optional, Literal, Union, Tuple

from haystack.schema import Document
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
        allowed_domains: Optional[List[str]] = None,
        link_content_fetcher: Optional[LinkContentFetcher] = None,
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
        :param allowed_domains: List of domains to restrict the search to. If not provided, the search is unrestricted.
        :param link_content_fetcher: LinkContentFetcher to be used to fetch the content from the links. If not provided,
        the default LinkContentFetcher is used.

        """
        super().__init__()
        self.web_search = WebSearch(
            api_key=api_key,
            top_k=top_search_results,
            allowed_domains=allowed_domains,
            search_engine_provider=search_engine_provider,
        )
        self.link_content_fetcher = link_content_fetcher or LinkContentFetcher()
        self.mode = mode
        self.cache_document_store = cache_document_store
        self.document_store = cache_document_store
        self.cache_index = cache_index
        self.top_k = top_k
        self.cache_headers = cache_headers
        self.cache_time = cache_time
        self.preprocessor = (
            preprocessor or PreProcessor(progress_bar=False) if mode == "preprocessed_documents" else None
        )

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
        for frequently accessed URLs.

        :param query: The query string.
        :param top_k: The number of Documents to be returned by the retriever.
        :param preprocessor: The PreProcessor to be used to split documents into paragraphs.
        :param cache_document_store: The DocumentStore to cache the documents to.
        :param cache_index: The index name to save the documents to.
        :param cache_headers: The headers to save the documents to.
        :param cache_time: The time limit in seconds for the documents in the cache. If objects are older than this time,
        they will be deleted from the cache on the next retrieval.
        """

        # Initialize default parameters
        preprocessor = preprocessor or self.preprocessor
        cache_index = cache_index or self.cache_index
        top_k = top_k or self.top_k
        cache_headers = cache_headers or self.cache_headers
        cache_time = cache_time or self.cache_time

        search_results, _ = self.web_search.run(query=query)
        result_docs = search_results["documents"]

        if self.mode != "snippets":
            # for raw_documents and preprocessed_documents modes, we need to retrieve the links from the search results
            links: List[SearchResult] = self._prepare_links(result_docs)

            links_found_in_cache, cached_docs = self._check_cache(links)
            logger.debug("Found %d links in cache", len(links_found_in_cache))

            links_to_fetch = [link for link in links if link not in links_found_in_cache]
            logger.debug("Fetching %d links", len(links_to_fetch))
            result_docs = self._scrape_links(links_to_fetch)

            # Save result_docs to cache
            self._save_to_cache(
                result_docs, cache_index=cache_index, cache_headers=cache_headers, cache_time=cache_time
            )

            # join cached_docs and result_docs
            result_docs = cached_docs + result_docs

            # Preprocess documents
            if preprocessor:
                result_docs = preprocessor.process(result_docs)

        # Return results
        return result_docs[:top_k]

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

    def _scrape_links(self, links: List[SearchResult]) -> List[Document]:
        """
        Scrape the links and return the documents.

        :param links: List of SearchResult objects.
        :return: List of Document objects obtained by fetching the content from the links.
        """
        if not links:
            return []

        def link_fetch(link: SearchResult) -> List[Document]:
            """
            Encapsulate the link fetching logic in a function to be used in a ThreadPoolExecutor.
            """
            docs: List[Document] = []
            try:
                docs = self.link_content_fetcher.fetch(
                    url=link.url,
                    doc_kwargs={
                        "id_hash_keys": ["meta.url"],
                        "search.score": link.score,
                        "search.position": link.position,
                        "snippet_text": link.snippet,
                    },
                )
            except Exception as e:
                # Log the exception for debugging
                logger.debug("Error fetching documents from %s : %s", link.url, str(e))

            return docs

        thread_count = min(cpu_count() if len(links) > cpu_count() else len(links), 10)  # max 10 threads
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            fetched_pages: Iterator[List[Document]] = executor.map(link_fetch, links)

        # Flatten list of lists to a single list
        extracted_docs = [doc for doc_list in fetched_pages for doc in doc_list]

        # Sort by score
        extracted_docs = sorted(extracted_docs, key=lambda x: x.meta["search.score"], reverse=True)

        return extracted_docs

    def _check_cache(self, links: List[SearchResult]) -> Tuple[List[SearchResult], List[Document]]:
        """
        Check the DocumentStore cache for documents.

        :param links: List of SearchResult objects.
        :return: Tuple of lists of SearchResult and Document objects that were found in the cache.
        """
        if not links or not self.cache_document_store:
            return [], []

        cache_documents: List[Document] = []
        cached_links: List[SearchResult] = []

        valid_links = [link for link in links if link.url]
        for link in valid_links:
            cache_filter: FilterType = {"url": link.url}
            documents = self.cache_document_store.get_all_documents(filters=cache_filter, return_embedding=False)
            if documents:
                cache_documents.extend(documents)
                cached_links.append(link)

        return cached_links, cache_documents

    def _save_to_cache(
        self,
        documents: List[Document],
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: Optional[int] = None,
    ) -> None:
        """
        Save the documents to the cache and potentially delete old expired documents from the cache.

        :param documents: List of Document objects to be saved to the cache.
        :param cache_index: Optional index name to save the documents to.
        :param cache_headers: Optional headers made to use when saving the documents to the cache.
        :param cache_time: Optional time to live in seconds for the documents in the cache. If objects are older than
        this time, they will be deleted from the cache.
        """
        cache_document_store = self.cache_document_store

        if cache_document_store is not None and documents:
            cache_document_store.write_documents(
                documents=documents, index=cache_index, headers=cache_headers, duplicate_documents="overwrite"
            )

        if cache_document_store and cache_time is not None and cache_time > 0:
            cache_filter: FilterType = {
                "timestamp": {"$lt": int((datetime.utcnow() - timedelta(seconds=cache_time)).timestamp())}
            }

            cache_document_store.delete_documents(index=cache_index, headers=cache_headers, filters=cache_filter)
            logger.debug("Deleted documents in the cache using filter: %s", cache_filter)

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
        :param cache_time: The time limit in seconds for the documents in the cache.

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
                )
            )

        return documents
