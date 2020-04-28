import logging
from collections import OrderedDict, namedtuple

import pandas as pd
from haystack.database.base import Document
from haystack.retriever.base import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)

# TODO make Paragraph generic for configurable units of text eg, pages, paragraphs, or split by a char_limit
Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "text"])


class TfidfRetriever(BaseRetriever):
    """
    Read all documents from a SQL backend.

    Split documents into smaller units (eg, paragraphs or pages) to reduce the 
    computations when text is passed on to a Reader for QA.

    It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.
    """

    def __init__(self, document_store):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
        )

        self.document_store = document_store
        self.paragraphs = self._get_all_paragraphs()
        self.df = None
        self.fit()

    def _get_all_paragraphs(self):
        """
        Split the list of documents in paragraphs
        """
        documents = self.document_store.get_all_documents()

        paragraphs = []
        p_id = 0
        for doc in documents:
            for p in doc.text.split("\n\n"):
                if not p.strip():  # skip empty paragraphs
                    continue
                paragraphs.append(
                    Paragraph(document_id=doc.id, paragraph_id=p_id, text=(p,))
                )
                p_id += 1
        logger.info(f"Found {len(paragraphs)} candidate paragraphs from {len(documents)} docs in DB")
        return paragraphs

    def _calc_scores(self, query):
        question_vector = self.vectorizer.transform([query])

        scores = self.tfidf_matrix.dot(question_vector.T).toarray()
        idx_scores = [(idx, score) for idx, score in enumerate(scores)]
        indices_and_scores = OrderedDict(
            sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)
        )
        return indices_and_scores

    def retrieve(self, query, filters=None, top_k=10, verbose=True):
        if filters:
            raise NotImplementedError("Filters are not implemented in TfidfRetriever.")

        # get scores
        indices_and_scores = self._calc_scores(query)

        print(indices_and_scores)

        # rank paragraphs
        df_sliced = self.df.loc[indices_and_scores.keys()]
        df_sliced = df_sliced[:top_k]

        if verbose:
            logger.info(
                f"Identified {df_sliced.shape[0]} candidates via retriever:\n {df_sliced.to_string(col_space=10, index=False)}"
            )

        # get actual content for the top candidates
        paragraphs = list(df_sliced.text.values)
        meta_data = [{"document_id": row["document_id"], "paragraph_id": row["paragraph_id"]}
                     for idx, row in df_sliced.iterrows()]

        documents = []
        for para, meta in zip(paragraphs, meta_data):
            documents.append(
                Document(
                    id=meta["paragraph_id"],
                    text=para
                ))

        return documents

    def fit(self):
        self.df = pd.DataFrame.from_dict(self.paragraphs)
        self.df["text"] = self.df["text"].apply(lambda x: " ".join(x))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])
