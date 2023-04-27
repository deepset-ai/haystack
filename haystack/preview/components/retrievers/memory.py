from typing import Optional, Dict, Any, List, Literal, Tuple, get_args

import re
import logging

import rank_bm25
import numpy as np
from tqdm import tqdm

from haystack.preview import Document, component
from haystack.preview.document_stores import MemoryDocumentStore
from haystack.preview.utils.math_utils import expit


logger = logging.getLogger(__name__)
RetrievalMethod = Literal["bm25"]


@component
class MemoryRetriever:
    """
    Retrieves documents from a MemoryDocumentStore.

    Supports BM25 Retrieval.
    """

    def __init__(
        self,
        input: str = "queries",
        output: str = "documents",
        document_store: str = "document_store",
        retrieve_by: RetrievalMethod = "bm25",
    ):
        self.inputs = [input]
        self.outputs = [output]
        self.document_store = document_store
        self.retrieve_by = retrieve_by

    def run(self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Dict[str, Any]]):
        """
        Performs the retrieval.

        It expects the following parameters:
        - the value of `self.document_store`: this entry should contain a `MemoryDocumentStore` instance. Attach such
            instance to the pipeline with `add_store` to make this happen automatically.
        - `retrieve_by`: to specify the retrieval method, by default `bm25`.
        - Any parameter that can be passed to `MemoryRetriever.retrieve_by_bm25()` except for `docstore` and `query`,
            like `filters`, `top_k`, `scale_score`, and so on. See `MemoryRetriever.retrieve_by_bm25()`.
        """
        params = parameters.get(name, {})

        for key in params.keys():
            if key not in [self.document_store, "retrieve_by", "bm25_parameters"]:
                logger.warning(f"MemoryRetriever received an unexpected parameter: '{key}'. It will be ignored.")

        docstore = params.pop(self.document_store, None)
        if not docstore or not isinstance(docstore, MemoryDocumentStore):
            raise ValueError("MemoryRetriever needs a MemoryDocumentStore. Add one to the Pipeline.")

        queries = [value for key, value in data if key == self.inputs[0]]

        retrieve_by = params.pop("retrieve_by", self.retrieve_by)
        if retrieve_by == "bm25":
            results = []
            for query in queries:
                documents = self.retrieve_by_bm25(docstore=docstore, query=query, **params)
                results.append(documents)
        else:
            raise ValueError(
                f"Retrieval method '{retrieve_by}' not supported. Supported methods: {get_args(RetrievalMethod)}"
            )

        return ({self.outputs[0]: results}, parameters)

    def retrieve_by_bm25(
        self,
        docstore: MemoryDocumentStore,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        scale_score: bool = True,
        progress_bar: bool = True,
        tokenization_regex: str = r"(?u)\b\w\w+\b",
        algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        **bm25_parameters,
    ) -> List[Document]:
        """
        Performs BM25 retrieval using `rank_bm25`.

        :param docstore: the document store to retrieve from
        :param query: the query, as a string
        :param filters: perform retrieval only on the subset defined by this filter
        :param top_k: how many hits to return. Note that it might return less than top_k if the store
            contains less than top_k documents, or the filters returnes less than top_k documents.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
        :param progress_bar: enables/disables progress bars for long operations.
        :param tokenization_regex: how to tokenize the text before ranking by BM25.
        :param algorithm: the BM25 algorithm to use (`BM25Okapi`, `BM25L` or `BM25Plus`, see
            [`rank_bm25` documentation](https://github.com/dorianbrown/rank_bm25))
        :param parameters: Any parameters to pass to the BM25 index when retrieval by BM25 is performed.
            Otherwise unused. see [`rank_bm25` documentation](https://github.com/dorianbrown/rank_bm25)

        :returns: a list of documents in order of relevance. The documents have the score field populated
            with the value computed by bm25 against the given query.
        """
        if not query:
            raise ValueError("The query can't empty.")

        documents = docstore.filter_documents(filters=filters)
        tokenizer = re.compile(tokenization_regex).findall

        # Tokenize the query
        tokenized_query = tokenizer(query.lower())

        # Tokenize the documents
        tokenized_corpus = []
        for doc in tqdm(documents, unit=" docs", desc="Updating BM25 ranking...", disable=not progress_bar):
            if doc.content_type != "text":
                logger.warning(
                    "Type of document %s is not 'text'. It won't be present in the BM25 ranking. "
                    "To silence this warning, consider using different document stores for different "
                    "document types, or switch to retrieval by embedding.",
                    doc.id,
                )
            else:
                tokenized_corpus.append(tokenizer(doc.content.lower()))

        # Rank all the documents
        algorithm = getattr(rank_bm25, algorithm)
        ranking = algorithm(tokenized_corpus, **bm25_parameters)
        docs_scores = ranking.get_scores(tokenized_query)

        # Scale the scores if nedded
        if scale_score is True:
            docs_scores = [float(expit(np.asarray(score / 8))) for score in docs_scores]

        # Sort the documents and return the top_k with their scores
        top_docs_positions = np.argsort(docs_scores)[::-1][:top_k]
        top_docs = []
        for i in top_docs_positions:
            doc = documents[i].to_dict()
            doc["score"] = docs_scores[i]
            top_docs.append(Document.from_dict(doc))

        return top_docs
