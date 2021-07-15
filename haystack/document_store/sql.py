import itertools
import logging
import collections
from typing import Any, Dict, Union, List, Optional, Generator
from uuid import uuid4

import numpy as np
from sqlalchemy import and_, func, create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import case, null

from haystack import Document, Label
from haystack.document_store.base import BaseDocumentStore

logger = logging.getLogger(__name__)


Base = declarative_base()  # type: Any


class ORMBase(Base):
    __abstract__ = True

    id = Column(String(100), default=lambda: str(uuid4()), primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), server_onupdate=func.now())


class DocumentORM(ORMBase):
    __tablename__ = "document"

    text = Column(Text, nullable=False)
    index = Column(String(100), nullable=False)
    vector_id = Column(String(100), unique=True, nullable=True)

    # speeds up queries for get_documents_by_vector_ids() by having a single query that returns joined metadata
    meta = relationship("MetaORM", back_populates="documents", lazy="joined")


class MetaORM(ORMBase):
    __tablename__ = "meta"

    name = Column(String(100), index=True)
    value = Column(String(1000), index=True)
    document_id = Column(
        String(100),
        ForeignKey("document.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True
    )

    documents = relationship(DocumentORM, back_populates="meta")


class LabelORM(ORMBase):
    __tablename__ = "label"

    document_id = Column(String(100), ForeignKey("document.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    index = Column(String(100), nullable=False)
    no_answer = Column(Boolean, nullable=False)
    origin = Column(String(100), nullable=False)
    question = Column(Text, nullable=False)
    is_correct_answer = Column(Boolean, nullable=False)
    is_correct_document = Column(Boolean, nullable=False)
    answer = Column(Text, nullable=False)
    offset_start_in_doc = Column(Integer, nullable=False)
    model_id = Column(Integer, nullable=True)


class SQLDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        url: str = "sqlite://",
        index: str = "document",
        label_index: str = "label",
        duplicate_documents: str = "overwrite"
    ):
        """
        An SQL backed DocumentStore. Currently supports SQLite, PostgreSQL and MySQL backends.

        :param url: URL for SQL database as expected by SQLAlchemy. More info here: https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
        :param index: The documents are scoped to an index attribute that can be used when writing, querying, or deleting documents. 
                      This parameter sets the default value for document index.
        :param label_index: The default value of index attribute for the labels.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
                url=url, index=index, label_index=label_index, duplicate_documents=duplicate_documents
        )

        engine = create_engine(url)
        ORMBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.index: str = index
        self.label_index = label_index
        self.duplicate_documents = duplicate_documents
        if getattr(self, "similarity", None) is None:
            self.similarity = None
        self.use_windowed_query = True
        if "sqlite" in url:
            import sqlite3
            if sqlite3.sqlite_version < "3.25":
                self.use_windowed_query = False

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        documents = self.get_documents_by_id([id], index)
        document = documents[0] if documents else None
        return document

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None, batch_size: int = 10_000) -> List[Document]:
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index

        documents = []
        for i in range(0, len(ids), batch_size):
            query = self.session.query(DocumentORM).filter(
                DocumentORM.id.in_(ids[i: i + batch_size]),
                DocumentORM.index == index
            )
            for row in query.all():
                documents.append(self._convert_sql_row_to_document(row))

        return documents

    def get_documents_by_vector_ids(self, vector_ids: List[str], index: Optional[str] = None, batch_size: int = 10_000):
        """Fetch documents by specifying a list of text vector id strings"""
        index = index or self.index

        documents = []
        for i in range(0, len(vector_ids), batch_size):
            query = self.session.query(DocumentORM).filter(
                DocumentORM.vector_id.in_(vector_ids[i: i + batch_size]),
                DocumentORM.index == index
            )
            for row in query.all():
                documents.append(self._convert_sql_row_to_document(row))

        sorted_documents = sorted(documents, key=lambda doc: vector_ids.index(doc.meta["vector_id"]))
        return sorted_documents

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        documents = list(self.get_all_documents_generator(index=index, filters=filters))
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """

        if return_embedding is True:
            raise Exception("return_embeddings is not supported by SQLDocumentStore.")
        result = self._query(
            index=index,
            filters=filters,
            batch_size=batch_size,
        )
        yield from result

    def _query(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        vector_ids: Optional[List[str]] = None,
        only_documents_without_embedding: bool = False,
        batch_size: int = 10_000
    ):
        """
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param vector_ids: List of vector_id strings to filter the documents by.
        :param only_documents_without_embedding: return only documents without an embedding.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        index = index or self.index
        # Generally ORM objects kept in memory cause performance issue
        # Hence using directly column name improve memory and performance.
        # Refer https://stackoverflow.com/questions/23185319/why-is-loading-sqlalchemy-objects-via-the-orm-5-8x-slower-than-rows-via-a-raw-my
        documents_query = self.session.query(
            DocumentORM.id,
            DocumentORM.text,
            DocumentORM.vector_id
        ).filter_by(index=index)

        if filters:
            documents_query = documents_query.join(MetaORM)
            for key, values in filters.items():
                documents_query = documents_query.filter(
                    MetaORM.name == key,
                    MetaORM.value.in_(values),
                    DocumentORM.id == MetaORM.document_id
                )
        if only_documents_without_embedding:
            documents_query = documents_query.filter(DocumentORM.vector_id.is_(None))
        if vector_ids:
            documents_query = documents_query.filter(DocumentORM.vector_id.in_(vector_ids))

        documents_map = {}

        if self.use_windowed_query:
            documents_query = self._windowed_query(documents_query, DocumentORM.id, batch_size)

        for i, row in enumerate(documents_query, start=1):
            documents_map[row.id] = Document(
                id=row.id,
                text=row.text,
                meta=None if row.vector_id is None else {"vector_id": row.vector_id}
            )
            if i % batch_size == 0:
                documents_map = self._get_documents_meta(documents_map)
                yield from documents_map.values()
                documents_map = {}
        if documents_map:
            documents_map = self._get_documents_meta(documents_map)
            yield from documents_map.values()

    def _get_documents_meta(self, documents_map):
        doc_ids = documents_map.keys()
        meta_query = self.session.query(
         MetaORM.document_id,
         MetaORM.name,
         MetaORM.value
        ).filter(MetaORM.document_id.in_(doc_ids))

        for row in meta_query.all():
            documents_map[row.document_id].meta[row.name] = row.value
        return documents_map

    def get_all_labels(self, index=None, filters: Optional[dict] = None):
        """
        Return all labels in the document store
        """
        index = index or self.label_index
        # TODO: Use batch_size
        label_rows = self.session.query(LabelORM).filter_by(index=index).all()
        labels = [self._convert_sql_row_to_label(row) for row in label_rows]

        return labels

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: add an optional index attribute to documents. It can be later used for filtering. For instance,
                      documents for evaluation can be indexed in a separate index than the documents for search.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.

        :return: None
        """

        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        if len(documents) == 0:
            return
        # Make sure we comply to Document class format
        if isinstance(documents[0], dict):
            document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
        else:
            document_objects = documents

        document_objects = self._handle_duplicate_documents(document_objects, duplicate_documents)
        for i in range(0, len(document_objects), batch_size):
            for doc in document_objects[i: i + batch_size]:
                meta_fields = doc.meta or {}
                vector_id = meta_fields.pop("vector_id", None)
                meta_orms = [MetaORM(name=key, value=value) for key, value in meta_fields.items()]
                doc_orm = DocumentORM(id=doc.id, text=doc.text, vector_id=vector_id, meta=meta_orms, index=index)
                if duplicate_documents == "overwrite":
                    # First old meta data cleaning is required
                    self.session.query(MetaORM).filter_by(document_id=doc.id).delete()
                    self.session.merge(doc_orm)
                else:
                    self.session.add(doc_orm)
            try:
                self.session.commit()
            except Exception as ex:
                logger.error(f"Transaction rollback: {ex.__cause__}")
                # Rollback is important here otherwise self.session will be in inconsistent state and next call will fail
                self.session.rollback()
                raise ex

    def write_labels(self, labels, index=None):
        """Write annotation labels into document store."""

        labels = [Label.from_dict(l) if isinstance(l, dict) else l for l in labels]
        index = index or self.label_index

        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(labels, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(f"Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                           f" This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                           f" the answer annotation and not the question."
                           f" Problematic ids: {','.join(duplicate_ids)}")
        # TODO: Use batch_size
        for label in labels:
            label_orm = LabelORM(
                id=label.id,
                document_id=label.document_id,
                no_answer=label.no_answer,
                origin=label.origin,
                question=label.question,
                is_correct_answer=label.is_correct_answer,
                is_correct_document=label.is_correct_document,
                answer=label.answer,
                offset_start_in_doc=label.offset_start_in_doc,
                model_id=label.model_id,
                index=index,
            )
            if label.id in duplicate_ids:
                self.session.merge(label_orm)
            else:
                self.session.add(label_orm)
        self.session.commit()

    def update_vector_ids(self, vector_id_map: Dict[str, str], index: Optional[str] = None, batch_size: int = 10_000):
        """
        Update vector_ids for given document_ids.

        :param vector_id_map: dict containing mapping of document_id -> vector_id.
        :param index: filter documents by the optional index attribute for documents in database.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        index = index or self.index
        for chunk_map in self.chunked_dict(vector_id_map, size=batch_size):
            self.session.query(DocumentORM).filter(
                DocumentORM.id.in_(chunk_map),
                DocumentORM.index == index
            ).update({
                DocumentORM.vector_id: case(
                    chunk_map,
                    value=DocumentORM.id,
                )
            }, synchronize_session=False)
            try:
                self.session.commit()
            except Exception as ex:
                logger.error(f"Transaction rollback: {ex.__cause__}")
                self.session.rollback()
                raise ex

    def reset_vector_ids(self, index: Optional[str] = None):
        """
        Set vector IDs for all documents as None
        """
        index = index or self.index
        self.session.query(DocumentORM).filter_by(index=index).update({DocumentORM.vector_id: null()})
        self.session.commit()

    def update_document_meta(self, id: str, meta: Dict[str, str]):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        self.session.query(MetaORM).filter_by(document_id=id).delete()
        meta_orms = [MetaORM(name=key, value=value, document_id=id) for key, value in meta.items()]
        for m in meta_orms:
            self.session.add(m)
        self.session.commit()

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index
        query = self.session.query(DocumentORM).filter_by(index=index)

        if filters:
            query = query.join(MetaORM)
            for key, values in filters.items():
                query = query.filter(MetaORM.name == key, MetaORM.value.in_(values))

        count = query.count()
        return count

    def get_label_count(self, index: Optional[str] = None) -> int:
        """
        Return the number of labels in the document store
        """
        index = index or self.label_index
        return self.session.query(LabelORM).filter_by(index=index).count()

    def _convert_sql_row_to_document(self, row) -> Document:
        document = Document(
            id=row.id,
            text=row.text,
            meta={meta.name: meta.value for meta in row.meta}
        )
        if row.vector_id:
            document.meta["vector_id"] = row.vector_id
        return document

    def _convert_sql_row_to_label(self, row) -> Label:
        label = Label(
            document_id=row.document_id,
            no_answer=row.no_answer,
            origin=row.origin,
            question=row.question,
            is_correct_answer=row.is_correct_answer,
            is_correct_document=row.is_correct_document,
            answer=row.answer,
            offset_start_in_doc=row.offset_start_in_doc,
            model_id=row.model_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            id=row.id
        )
        return label

    def query_by_embedding(self,
                           query_emb: np.ndarray,
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:

        raise NotImplementedError("SQLDocumentStore is currently not supporting embedding queries. "
                                  "Change the query type (e.g. by choosing a different retriever) "
                                  "or change the DocumentStore (e.g. to ElasticsearchDocumentStore)")

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """
        logger.warning(
                """DEPRECATION WARNINGS: 
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, filters)

    def delete_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """
        index = index or self.index

        if filters:
            # documents_query = documents_query.join(MetaORM)
            document_ids_to_delete = self.session.query(DocumentORM.id).filter_by(index=index)
            for key, values in filters.items():
                document_ids_to_delete = document_ids_to_delete.filter(
                        MetaORM.name == key,
                        MetaORM.value.in_(values),
                        DocumentORM.id == MetaORM.document_id
                )
            self.session.query(DocumentORM).filter(DocumentORM.id.in_(document_ids_to_delete)).delete(
                    synchronize_session=False)
        else:
            self.session.query(DocumentORM).filter_by(index=index).delete(synchronize_session=False)

        self.session.commit()

    def _get_or_create(self, session, model, **kwargs):
        instance = session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance
        else:
            instance = model(**kwargs)
            session.add(instance)
            session.commit()
            return instance

    def chunked_dict(self, dictionary, size):
        it = iter(dictionary)
        for i in range(0, len(dictionary), size):
            yield {k: dictionary[k] for k in itertools.islice(it, size)}

    def _column_windows(self, session, column, windowsize):
        """Return a series of WHERE clauses against
        a given column that break it into windows.

        Result is an iterable of tuples, consisting of
        ((start, end), whereclause), where (start, end) are the ids.

        The code is taken from: https://github.com/sqlalchemy/sqlalchemy/wiki/RangeQuery-and-WindowedRangeQuery
        """

        def int_for_range(start_id, end_id):
            if end_id:
                return and_(
                    column >= start_id,
                    column < end_id
                )
            else:
                return column >= start_id

        q = session.query(
            column,
            func.row_number(). \
                over(order_by=column). \
                label('rownum')
        ). \
            from_self(column)
        if windowsize > 1:
            q = q.filter(text("rownum %% %d=1" % windowsize))

        intervals = [id for id, in q]

        while intervals:
            start = intervals.pop(0)
            if intervals:
                end = intervals[0]
            else:
                end = None
            yield int_for_range(start, end)

    def _windowed_query(self, q, column, windowsize):
        """"Break a Query into windows on a given column."""

        for whereclause in self._column_windows(
                q.session,
                column, windowsize):
            for row in q.filter(whereclause).order_by(column):
                yield row
