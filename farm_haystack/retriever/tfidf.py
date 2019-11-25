from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from farm_haystack.database import db
from farm_haystack.database.orm import Document

logger = logging.getLogger(__name__)

# TODO make Paragraph generic for configurable units of text eg, pages, paragraphs, or split by a char_limit
Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "text"])


class BaseRetriever(ABC):
    @abstractmethod
    def _get_all_paragraphs(self):
        pass

    @abstractmethod
    def retrieve(self, query, candidate_doc_ids=None, top_k=1):
        pass

    @abstractmethod
    def fit(self):
        pass


class TfidfRetriever(BaseRetriever):
    """
    Read all documents from a SQL backend.

    Split documents into smaller units (eg, paragraphs or pages) to reduce the 
    computations when text is passed on to a Reader for QA.

    It uses sklearn TfidfVectorizer to compute a tf-idf matrix.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
        )

        self.paragraphs = self._get_all_paragraphs()
        self.df = None
        self.fit()

    def _get_all_paragraphs(self):
        """
        Split the list of documents in paragraphs
        """
        documents = db.session.query(Document).all()

        paragraphs = []
        p_id = 0
        for doc in documents:
            _pgs = [d for d in doc.text.splitlines() if d.strip()]
            for p in doc.text.split("\n\n"):
                if not p.strip():  # skip empty paragraphs
                    continue
                paragraphs.append(
                    Paragraph(document_id=doc.id, paragraph_id=p_id, text=(p,))
                )
                p_id += 1
        logger.info(f"Found {len(paragraphs)} candidate paragraphs from {len(documents)} docs in DB")
        return paragraphs

    def retrieve(self, query, candidate_doc_ids=None, top_k=10):
        question_vector = self.vectorizer.transform([query])

        scores = self.tfidf_matrix.dot(question_vector.T).toarray()
        idx_scores = [(idx, score) for idx, score in enumerate(scores)]
        top_k_scores = OrderedDict(
            sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)[:top_k]
        )
        return top_k_scores

    def fit(self):
        self.df = pd.DataFrame.from_dict(self.paragraphs)
        self.df["text"] = self.df["text"].apply(lambda x: " ".join(x))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])
