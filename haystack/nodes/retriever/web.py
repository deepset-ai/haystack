import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Literal
from unicodedata import combining, normalize

import mmh3
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


class WebRetriever(BaseRetriever):
    def __init__(
        self,
        web_search: WebSearch,
        top_p: Optional[float] = 0.95,
        mode: Literal["snippet", "document"] = "document",
        preprocessor: Optional[PreProcessor] = None,
        document_store: Optional[BaseDocumentStore] = None,
        document_index: Optional[str] = None,
        document_headers: Optional[Dict[str, str]] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: int = 24 * 60 * 60,
    ):
        """
        Collect complete documents from the web using the links provided by a WebSearch node
        """
        super().__init__()
        self.web_search = web_search
        self.mode = mode
        self.preprocessor = preprocessor
        self.document_store = document_store
        self.document_index = document_index
        self.document_headers = document_headers
        self.cache_document_store = cache_document_store
        self.cache_index = cache_index
        self.cache_headers = cache_headers
        self.cache_time = cache_time
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
        documents = []
        if self.cache_document_store is not None:
            query_norm = self._normalize_query(query)
            cache_filter: FilterType = {"$and": {"search.query": query_norm}}

            if cache_time is not None and cache_time > 0:
                cache_filter["timestamp"] = {
                    "$gt": int((datetime.utcnow() - timedelta(seconds=cache_time)).timestamp())
                }
                logger.debug("Cache filter: %s", cache_filter)

            documents = self.cache_document_store.get_all_documents(
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
            documents = [self._rebuild_cache_document(document) for document in documents]

            cache_document_store.write_documents(
                documents=documents, index=cache_index, headers=cache_headers, duplicate_documents="overwrite"
            )

            logger.debug("Saved %d documents in the cache", len(documents))

            cache_filter: FilterType = {"$and": {"query": query}}

            if cache_time is not None and cache_time > 0:
                cache_filter["timestamp"] = {
                    "$lt": int((datetime.utcnow() - timedelta(seconds=cache_time)).timestamp())
                }

                cache_document_store.delete_documents(index=cache_index, headers=cache_headers, filters=cache_filter)

                logger.debug("Deleted documents in the cache using filter: %s", cache_filter)

            return True

        return False

    def _save_documents(
        self, documents: List[Document], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        logger.debug("Saving %d documents", len(documents))

        if self.document_store is not None:
            documents = [self._rebuild_document(document) for document in documents]

            self.document_store.write_documents(
                documents=documents, index=index, headers=headers, duplicate_documents="overwrite"
            )

            return True

        return False

    def _rebuild_document(self, document: Document) -> Document:
        final_hash_key = document.content

        if document.meta:
            if "url" in document.meta:
                final_hash_key += ":" + document.meta["url"]
            if "split_id" in document.meta:
                final_hash_key += ":" + document.meta["split_id"]

        document.id = "{:02x}".format(mmh3.hash128(final_hash_key, signed=False))

        return document

    def _rebuild_cache_document(self, document: Document) -> Document:
        final_hash_key = "cache:"
        if document.meta:
            if "query" in document.meta:
                final_hash_key += document.meta["query"]
            if "url" in document.meta:
                final_hash_key += ":" + document.meta["url"]
            if "split_id" in document.meta:
                final_hash_key += ":" + document.meta["split_id"]
        else:
            final_hash_key = document.content

        document.id = "{:02x}".format(mmh3.hash128(final_hash_key, signed=False))

        return document

    def retrieve(
        self,
        query: str,
        top_p: Optional[int] = None,
        preprocessor: Optional[PreProcessor] = None,
        document_store: Optional[BaseDocumentStore] = None,
        document_index: Optional[str] = None,
        document_headers: Optional[Dict[str, str]] = None,
        cache_document_store: Optional[BaseDocumentStore] = None,
        cache_index: Optional[str] = None,
        cache_headers: Optional[Dict[str, str]] = None,
        cache_time: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve documents based on a WebSearchEngine list of URLs. The documents are scraped from the web at real-time.
        The documents can then stored in a DocumentStore for later use. The documents can be cached in a DocumentStore to improve
        retrieval time.
        :param query: The query string.
        :top_p: The top-p sampling parameter. If None, the default value is used.
        :index_name: The index name to save the documents to.
        :duplicate_documents: If "skip", documents with the same ID are skipped. If "overwrite", documents with the same ID are overwritten. If "fail", an exception is raised.
        use_cache: If True, the results are cached in the DocumentStore.
        cache_time: The time limit (seconds) to check the cache. Default is 24 hours.
        """
        if preprocessor is None:
            preprocessor = self.preprocessor
        if document_store is None:
            document_store = self.document_store
        if document_index is None:
            document_index = self.document_index
        if document_headers is None:
            document_headers = self.document_headers
        if cache_document_store is None:
            cache_document_store = self.cache_document_store
        if cache_index is None:
            cache_index = self.cache_index
        if cache_headers is None:
            cache_headers = self.cache_headers
        if cache_time is None:
            cache_time = self.cache_time

        query_norm = self._normalize_query(query)
        if cache_document_store:
            documents = self._check_cache(
                query_norm, cache_index=cache_index, cache_headers=cache_headers, cache_time=cache_time
            )
            if documents and len(documents) > 0:
                return documents

        search_results, _ = self.web_search.run(query=query)
        if self.sampler and search_results["documents"]:
            search_results, _ = self.sampler.run(query, search_results["documents"], top_p=top_p)
        search_results = search_results["documents"]
        if self.mode == "snippet":
            return search_results

        links: List[Tuple[str, Union[str, None], Union[str, None]]] = [
            (
                r.meta["link"],
                r.meta["score"] if r.meta.get("score") else None,
                r.meta["position"] if r.meta.get("position") else None,
            )
            for r in search_results
            if r.meta.get("link")
        ]
        extracted_docs = []

        def scrape_direct(link) -> Dict[str, Any]:
            try:
                response = fetch_url(link[0])

                if response is None:
                    logger.debug("No response from URL %s, trying Google Cache", link[0])
                    response = fetch_url(f"https://webcache.googleusercontent.com/search?q=cache:{link[0]}")

                if response is not None:
                    extracted = bare_extraction(
                        response,
                        include_comments=False,
                        include_tables=False,
                        include_links=False,
                        deduplicate=True,
                        date_extraction_params={"extensive_search": True, "outputformat": "%Y-%m-%d"},
                    )

                    if extracted is not None and isinstance(extracted, dict):
                        extracted["url"] = clean_url(link[0])
                        extracted["search.score"] = link[1]
                        extracted["search.position"] = link[2]

                        return {
                            k: v
                            for k, v in extracted.items()
                            if k not in ["fingerprint", "license", "body", "comments", "raw_text", "commentsbody"]
                        }

                return {}

            except Exception as e:
                logger.error("Error retrieving URL %s: %s", link[0], e)
                return {}

        thread_count = cpu_count() if len(links) > cpu_count() else len(links)
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results: Iterator[Dict[str, Any]] = executor.map(scrape_direct, links)

            failed = 0
            extracted_docs = []
            for r, s in zip(results, search_results):
                if r is not None and "text" in r:
                    r["id_hash_keys"] = ["content", "meta"]

                    document = Document.from_dict(r, field_map={"text": "content"})

                    document.meta["timestamp"] = int(datetime.utcnow().timestamp())
                    document.meta["search.position"] = s.meta.get("position")
                    document.meta["search.score"] = s.meta.get("score")
                    document.meta["search.query"] = query_norm

                    extracted_docs.append(document)
                else:
                    logger.warning("Could not extract text from URL %s. Using search snippet.", s.meta["link"])

                    if "date" in s.meta:
                        s.meta["date"] = find_date(s.meta["date"], extensive_search=True, outputformat="%Y-%m-%d")

                    if "link" in s.meta:
                        s.meta["url"] = clean_url(s.meta["link"])
                        del s.meta["link"]

                    score = s.meta.get("score")
                    if score is not None:
                        del s.meta["score"]

                    position = s.meta.get("position")
                    if position is not None:
                        del s.meta["position"]

                    s.meta["id_hash_keys"] = ["content", "meta"]
                    s.meta["search.query"] = query_norm

                    s.meta["search.position"] = position
                    s.meta["search.score"] = score
                    s.meta["search.snippet"] = 1
                    s.meta["timestamp"] = int(datetime.utcnow().timestamp())

                    extracted_docs.append(s)
                    failed += 1

            logger.debug(
                "Extracted %d documents / %s snippet from %s URLs.", len(extracted_docs) - failed, failed, len(links)
            )

        processed_docs = []

        if preprocessor is not None:
            processed_docs.extend(
                [
                    t
                    for d in extracted_docs
                    for t in preprocessor.process(
                        [d],
                        clean_whitespace=True,
                        clean_empty_lines=True,
                        split_by=preprocessor.split_by if preprocessor.split_by is not None else "words",  # type: ignore
                        split_length=preprocessor.split_length if preprocessor.split_length is not None else 180,
                        split_overlap=preprocessor.split_overlap if preprocessor.split_overlap is not None else 20,
                        split_respect_sentence_boundary=preprocessor.split_respect_sentence_boundary
                        if preprocessor.split_respect_sentence_boundary is not None
                        else False,
                    )
                ]
            )
        else:
            processed_docs = extracted_docs

        logger.debug("Processed %d documents resulting in %s documents", len(extracted_docs), len(processed_docs))

        if cache_document_store is not None:
            cached = self._save_cache(query_norm, processed_docs, cache_index=cache_index, cache_headers=cache_headers)
            if not cached:
                logger.warning(
                    "Could not save documents to cache document store. Please check your document store configuration."
                )

        if document_store is not None:
            saved = self._save_documents(documents=processed_docs, index=document_index, headers=document_headers)
            if not saved:
                logger.warning(
                    "Could not save documents to document store. Please check your document store configuration."
                )

        return processed_docs

    def retrieve_batch(
        self,
        queries: List[str],
        top_p: Optional[int] = None,
        preprocessor: Optional[PreProcessor] = None,
        document_store: Optional[BaseDocumentStore] = None,
        document_index: Optional[str] = None,
        document_headers: Optional[Dict[str, str]] = None,
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
                    document_store=document_store,
                    document_index=document_index,
                    document_headers=document_headers,
                    cache_document_store=cache_document_store,
                    cache_index=cache_index,
                    cache_headers=cache_headers,
                    cache_time=cache_time,
                )
            )

        return documents
