import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from typing import Any, Dict, Iterator, List, Optional, Literal, Union
from unicodedata import combining, normalize

import requests
from boilerpy3 import extractors

from haystack import Document, __version__
from haystack.document_stores.base import BaseDocumentStore
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.search_engine.web import SearchEngine
from haystack.nodes.search_engine.web import WebSearch
from haystack.schema import FilterType

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    url: str
    score: Optional[str]
    position: Optional[str]


class WebRetriever(BaseRetriever):
    """
    WebRetriever makes it possible to query the web for relevant documents. It downloads web page results returned by WebSearch, strips HTML, and extracts raw text, which is then
    split into smaller documents using the optional PreProcessor.

    WebRetriever operates in two modes:

    - snippets mode: WebRetriever returns a list of Documents. Each Document is a snippet of the search result.
    - raw_documents mode: WebRetriever returns a list of Documents. Each Document is a full website returned by the search, stripped of HTML.
    - preprocessed_documents mode: WebRetriever return a list of Documents. Each Document is a preprocessed split of the full website stripped of HTML.

    In the preprocessed_documents mode, after WebSearch receives the query through the `run()` method, it fetches the top_k URLs relevant to the query. WebSearch then downloads and processes these URLs.
    The processing involves stripping HTML tags and producing
    a clean, raw text wrapped in the Document objects. WebRetriever then splits raw text into Documents according to the PreProcessor settings.
    Finally, WebRetriever returns the top_k preprocessed Documents.

    Finding the right balance between top_k and top_p is crucial to obtain high-quality and diverse results in the document
    mode. To explore potential results, we recommend that you set top_k for WebSearch close to 10.
    However, keep in mind that setting a high top_k value results in fetching and processing numerous web pages and is heavier on the resources.

    We recommend you use the default value for top_k and adjust it based on your specific
    use case. The default value is 5. This means WebRetriever returns at most
    five of the most relevant processed documents, ensuring the search results are diverse but still of high
    quality. To get more results, increase top_k.
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
        :param top_k: Top k documents to be returned by the retriever.
        :param mode: Whether to return snippets, raw documents, or preprocessed documents. Preprocessed documents are the default.
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
        self.cache_index = cache_index
        self.cache_headers = cache_headers
        self.cache_time = cache_time
        self.top_k = top_k
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
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
        """Check documents retrieved based on the query in cache."""

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
        Retrieve documents based on the list of URLs from the WebSearchEngine. The documents are scraped from the web
        at real-time. You can then store the documents in a DocumentStore for later use. You can cache them in a
        DocumentStore to improve retrieval time.
        :param query: The query string.
        :param top_k: The number of documents to be returned by the retriever. If None, the default value is used.
        :param preprocessor: The PreProcessor to be used to split documents into paragraphs.
        :param cache_document_store: The DocumentStore to cache the documents to.
        :param cache_index: The index name to save the documents to.
        :param cache_headers: The headers to save the documents to.
        :param cache_time: The time limit in seconds to check the cache. The default is 24 hours.
        """

        preprocessor = preprocessor or self.preprocessor
        cache_document_store = cache_document_store or self.cache_document_store
        cache_index = cache_index or self.cache_index
        cache_headers = cache_headers or self.cache_headers
        cache_time = cache_time or self.cache_time
        top_k = top_k or self.top_k

        query_norm = self._normalize_query(query)

        extracted_docs = self._check_cache(
            query_norm, cache_index=cache_index, cache_headers=cache_headers, cache_time=cache_time
        )

        # cache miss
        if not extracted_docs:
            search_results, _ = self.web_search.run(query=query)
            search_results = search_results["documents"]
            if self.mode == "snippets":
                return search_results  # type: ignore

            links: List[SearchResult] = [
                SearchResult(r.meta["link"], r.meta.get("score", None), r.meta.get("position", None))
                for r in search_results
                if r.meta.get("link")
            ]
            logger.debug("Starting to fetch %d links from WebSearch results", len(links))

            def scrape_direct(link: SearchResult) -> Dict[str, Any]:
                extractor = extractors.ArticleExtractor(raise_on_failure=False)
                try:
                    extracted_doc = {}
                    response = requests.get(link.url, headers=self._request_headers(), timeout=10)
                    if response.status_code == 200 and len(response.text) > 0:
                        extracted_content = extractor.get_content(response.text)
                        if extracted_content:
                            extracted_doc = {
                                "text": extracted_content,
                                "url": link.url,
                                "search.score": link.score,
                                "search.position": link.position,
                            }
                    return extracted_doc

                except Exception as e:
                    logger.error("Error retrieving URL %s: %s", link.url, e)
                    return {}

            thread_count = cpu_count() if len(links) > cpu_count() else len(links)
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                scraped_pages: Iterator[Dict[str, Any]] = executor.map(scrape_direct, links)

                failed = 0
                extracted_docs = []
                for scraped_page, search_result_doc in zip(scraped_pages, search_results):
                    if scraped_page and "text" in scraped_page:
                        document = self._document_from_scraped_page(search_result_doc, scraped_page, query_norm)
                        extracted_docs.append(document)
                    else:
                        logger.debug(
                            "Could not extract text from URL %s. Using search snippet.", search_result_doc.meta["link"]
                        )
                        snippet_doc = self._document_from_snippet(search_result_doc, query_norm)
                        extracted_docs.append(snippet_doc)
                        failed += 1

                logger.debug(
                    "Extracted %d documents / %s snippets from %s URLs.",
                    len(extracted_docs) - failed,
                    failed,
                    len(links),
                )

        if cache_document_store:
            cached = self._save_cache(query_norm, extracted_docs, cache_index=cache_index, cache_headers=cache_headers)
            if not cached:
                logger.warning(
                    "Could not save retrieved documents to the DocumentStore cache. "
                    "Check your document store configuration."
                )

        processed_docs = (
            [t for d in extracted_docs for t in preprocessor.process([d])]
            if self.mode == "preprocessed_documents"
            else extracted_docs
        )

        logger.debug("Processed %d documents resulting in %s documents", len(extracted_docs), len(processed_docs))
        return processed_docs[:top_k]

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
        documents = []

        # TODO: parallelize using ProcessPoolExecutor and use Lock at document store methods
        for q in queries:
            documents.extend(
                self.retrieve(
                    q,
                    top_p=top_p,
                    top_k=top_k,
                    preprocessor=preprocessor,
                    cache_document_store=cache_document_store,
                    cache_index=cache_index,
                    cache_headers=cache_headers,
                    cache_time=cache_time,
                )
            )

        return [documents]

    def _request_headers(self):
        headers = {
            "accept": "*/*",
            "User-Agent": f"haystack/WebRetriever/{__version__}",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
            "referer": "https://www.google.com/",
        }
        return headers

    def _document_from_snippet(self, doc, query_norm):
        doc_dict = {
            "text": doc.content,
            "url": doc.meta["link"],
            "id_hash_keys": ["meta.url", "meta.search.query"],
            "search.query": query_norm,
        }
        d = Document.from_dict(doc_dict, field_map={"text": "content"})
        d.meta["timestamp"] = int(datetime.utcnow().timestamp())
        d.meta["search.position"] = doc.meta.pop("position", None)
        d.meta["search.snippet"] = 1
        return d

    def _document_from_scraped_page(self, search_result_doc, scraped_page, query_norm):
        scraped_page["id_hash_keys"] = ["meta.url", "meta.search.query"]
        scraped_page["search.query"] = query_norm
        scraped_page.pop("description", None)
        document = Document.from_dict(scraped_page, field_map={"text": "content"})
        document.meta["timestamp"] = int(datetime.utcnow().timestamp())
        document.meta["search.position"] = search_result_doc.meta.get("position")
        return document
