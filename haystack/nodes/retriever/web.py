import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from typing import Any, Dict, Iterator, List, Optional, Literal
from unicodedata import combining, normalize

from courlan.clean import clean_url
from htmldate.core import find_date
from trafilatura import bare_extraction
from trafilatura.downloads import fetch_url

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
    def __init__(
        self,
        web_search: WebSearch,
        top_p: Optional[float] = 0.95,
        mode: Literal["snippet", "document"] = "document",
        preprocessor: Optional[PreProcessor] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: int = 2 * 24 * 60 * 60,
        apply_sampler_to_processed_docs: Optional[bool] = True,
    ):
        """
        Collect complete documents from the web using the links from the WebSearch node.
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
        if top_p is not None:
            self.sampler = TopPSampler(top_p=top_p, top_score_name="score")

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
        top_p: Optional[int] = None,
        preprocessor: Optional[PreProcessor] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: Optional[int] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieve documents based on the list of URLs from the WebSearchEngine. The documents are scraped from the web at real-time.
        You can then store the documents in a DocumentStore for later use. They can be cached in a DocumentStore to improve
        retrieval time.
        :param query: The query string.
        :top_p: The top-p sampling parameter that helps to strike a balance between the coherence and diversity of the generated text. If you need factual answers, set it to a lower value, like `0`. If set to `None`, the default value is used.
        :index_name: The index name to save the documents to.
        :duplicate_documents: Determines how to handle documents with the same ID. If set to "skip", it skips documents with the same ID. If set to "overwrite", it overwrites documents with the same ID. If set to "fail", it raises an exception.
        use_cache: If set to `True`, it caches the results in the DocumentStore.
        cache_time: The time limit in seconds to check the cache. The default is 24 hours.
        """

        preprocessor = preprocessor or self.preprocessor
        cache_document_store = cache_document_store or self.cache_document_store
        cache_index = cache_index or self.cache_index
        cache_headers = cache_headers or self.cache_headers
        cache_time = cache_time or self.cache_time

        query_norm = self._normalize_query(query)

        extracted_docs = self._check_cache(
            query_norm, cache_index=cache_index, cache_headers=cache_headers, cache_time=cache_time
        )

        # cache miss
        if not extracted_docs:
            search_results, _ = self.web_search.run(query=query)

            if self.sampler and search_results["documents"]:
                search_results, _ = self.sampler.run(query, search_results["documents"], top_p=top_p)
            search_results = search_results["documents"]
            if self.mode == "snippet":
                return search_results  # type: ignore

            links: List[SearchResult] = [
                SearchResult(r.meta["link"], r.meta.get("score", None), r.meta.get("position", None))
                for r in search_results
                if r.meta.get("link")
            ]

            def scrape_direct(link: SearchResult) -> Dict[str, Any]:
                try:
                    response = fetch_url(link.url)

                    if response is None:
                        logger.debug("No response from URL %s, trying Google cache", link.url)
                        response = fetch_url(f"https://webcache.googleusercontent.com/search?q=cache:{link.url}")

                    if response is not None:
                        extracted = bare_extraction(
                            response,
                            include_comments=False,
                            include_tables=False,
                            include_links=False,
                            deduplicate=True,
                            date_extraction_params={
                                "extensive_search": True,
                                "outputformat": "%Y-%m-%d",
                                "original_date": True,
                            },
                        )

                        if isinstance(extracted, dict):
                            extracted["url"] = clean_url(link.url)
                            extracted["search.score"] = link.score
                            extracted["search.position"] = link.position

                            return {
                                k: v
                                for k, v in extracted.items()
                                if k not in ["fingerprint", "license", "body", "comments", "raw_text", "commentsbody"]
                            }

                    return {}

                except Exception as e:
                    logger.error("Error retrieving URL %s: %s", link.url, e)
                    return {}

            thread_count = cpu_count() if len(links) > cpu_count() else len(links)
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                results: Iterator[Dict[str, Any]] = executor.map(scrape_direct, links)

                failed = 0
                extracted_docs = []
                for r, doc in zip(results, search_results):
                    if r and "text" in r:
                        r["id_hash_keys"] = ["meta.url", "meta.search.query"]
                        r["search.query"] = query_norm

                        r.pop("description", None)
                        document = Document.from_dict(r, field_map={"text": "content"})

                        document.meta["timestamp"] = int(datetime.utcnow().timestamp())
                        document.meta["search.position"] = doc.meta.get("position")
                        document.meta["search.score"] = doc.meta.get("score")

                        extracted_docs.append(document)
                    else:
                        logger.warning("Could not extract text from URL %s. Using search snippet.", doc.meta["link"])

                        if "date" in doc.meta:
                            doc.meta["date"] = find_date(
                                doc.meta["date"], extensive_search=True, outputformat="%Y-%m-%d"
                            )

                        if "link" in doc.meta:
                            doc.meta["url"] = clean_url(doc.meta["link"])
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
                    "Could not save documents to the DocumentStore cache. Check your document store configuration."
                )

        processed_docs = (
            [t for d in extracted_docs for t in preprocessor.process([d])] if preprocessor else extracted_docs
        )

        logger.debug("Processed %d documents resulting in %s documents", len(extracted_docs), len(processed_docs))

        if self.sampler and self.apply_sampler_to_processed_docs:
            sampled_docs, _ = self.sampler.run(query, processed_docs)
            if sampled_docs is not None:
                processed_docs = sampled_docs["documents"]

        return processed_docs if processed_docs else []

    def retrieve_batch(  # type: ignore[override]
        self,
        queries: List[str],
        top_p: Optional[int] = None,
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
                    preprocessor=preprocessor,
                    cache_document_store=cache_document_store,
                    cache_index=cache_index,
                    cache_headers=cache_headers,
                    cache_time=cache_time,
                )
            )

        return [documents]
