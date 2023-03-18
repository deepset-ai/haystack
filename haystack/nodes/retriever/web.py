import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from typing import Any, Dict, Iterator, List, Optional, Literal
from unicodedata import combining, normalize

import requests
from htmldate.core import find_date
from boilerpy3 import extractors

from haystack import Document
from haystack.document_stores.base import BaseDocumentStore
from haystack.nodes import TopPSampler
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.retriever.base import BaseRetriever
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
    WebRetriever is a retriever allowing users to query the web for relevant documents.

    WebRetriever operates in two modes:

    - snippet mode: WebRetriever will return a list of Documents, each Document being a snippet of the search result
    - document mode: WebRetriever will return a list of Documents, each Document being a full HTML stripped document
    of the search result

    In document mode, given a user query passed via the run method, WebSearch first fetches the top k query relevant
    URL results, which are in turn downloaded and processed. The processing involves stripping HTML tags and producing
    clean raw text wrapped in Document(s). WebRetriever then splits raw text into Documents of the desired preprocessor
    specified size. Finally, WebRetriever applies top p sampling on these Documents and returns at most top_k Documents.

    """

    def __init__(
        self,
        web_search: WebSearch,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = 5,
        mode: Literal["snippet", "document"] = "document",
        preprocessor: Optional[PreProcessor] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: int = 1 * 24 * 60 * 60,
        apply_sampler_to_processed_docs: Optional[bool] = True,
    ):
        """
        :param web_search: WebSearch node.
        :param top_p: Top p to apply to the retrieved and processed documents.
        :param top_k: Top k documents to be returned by the retriever.
        :param mode: Whether to return snippets or full documents.
        :param preprocessor: Preprocessor to be used to split documents into paragraphs.
        :param cache_document_store: DocumentStore to be used to cache search results.
        :param cache_index: Index name to be used to cache search results.
        :param cache_headers: Headers to be used to cache search results.
        :param cache_time: Time in seconds to cache search results. Defaults to 24 hours.
        :param apply_sampler_to_processed_docs: Whether to apply sampler to processed documents. If True, the sampler
        will be applied to the processed documents.
        """
        super().__init__()
        self.web_search = web_search
        self.mode = mode
        self.preprocessor = preprocessor
        self.cache_document_store = cache_document_store
        self.cache_index = cache_index
        self.cache_headers = cache_headers
        self.cache_time = cache_time
        self.apply_sampler_to_processed_docs = apply_sampler_to_processed_docs
        self.top_k = top_k
        if top_p is not None:
            self.sampler = TopPSampler(top_p=top_p, top_score_field="score")

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
        top_p: Optional[float] = None,
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
        at real-time. You can then store the documents in a DocumentStore for later use. They can be cached in a
        DocumentStore to improve retrieval time.
        :param query: The query string
        :param top_p: The top-p sampling parameter to be used to sample documents. If None, the default value is used.
        :param top_k: The number of documents to be returned by the retriever. If None, the default value is used.
        :param preprocessor: The preprocessor to be used to split documents into paragraphs.
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
            if self.mode == "snippet":
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
                    extracted_content = ""
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
                for scraped_page, doc in zip(scraped_pages, search_results):
                    if scraped_page and "text" in scraped_page:
                        scraped_page["id_hash_keys"] = ["meta.url", "meta.search.query"]
                        scraped_page["search.query"] = query_norm

                        scraped_page.pop("description", None)
                        document = Document.from_dict(scraped_page, field_map={"text": "content"})

                        document.meta["timestamp"] = int(datetime.utcnow().timestamp())
                        document.meta["search.position"] = doc.meta.get("position")
                        document.meta["search.score"] = doc.meta.get("score")

                        extracted_docs.append(document)
                    else:
                        logger.debug("Could not extract text from URL %s. Using search snippet.", doc.meta["link"])

                        if "date" in doc.meta:
                            doc.meta["date"] = find_date(
                                doc.meta["date"], extensive_search=True, outputformat="%Y-%m-%d"
                            )

                        if "link" in doc.meta:
                            doc.meta["url"] = doc.meta["link"]
                            del doc.meta["link"]

                        doc.meta["id_hash_keys"] = ["meta.url", "meta.search.query"]
                        doc.meta["search.query"] = query_norm

                        doc._get_id(id_hash_keys=doc.meta["id_hash_keys"])

                        doc.meta["search.position"] = doc.meta.pop("position", None)
                        doc.meta["search.score"] = doc.meta.pop("score", None)
                        doc.meta["search.snippet"] = 1
                        doc.meta["timestamp"] = int(datetime.utcnow().timestamp())

                        extracted_docs.append(doc)
                        failed += 1

                logger.debug(
                    "Extracted %d documents / %s snippet from %s URLs.",
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
            [t for d in extracted_docs for t in preprocessor.process([d])] if preprocessor else extracted_docs
        )

        logger.debug("Processed %d documents resulting in %s documents", len(extracted_docs), len(processed_docs))

        if self.sampler and self.apply_sampler_to_processed_docs:
            sampled_docs, _ = self.sampler.run(query, processed_docs)
            if sampled_docs is not None:
                processed_docs = sampled_docs["documents"]

        final_docs = processed_docs if processed_docs else []
        return final_docs[:top_k]

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
            "User-Agent": "haystack/WebRetriever/1.15",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
            "referer": "https://www.google.com/",
        }
        return headers
