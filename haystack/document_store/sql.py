import logging
from typing import Any, Dict, Union, List, Optional
from uuid import uuid4

from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import case

from haystack.document_store.base import BaseDocumentStore
from haystack import Document, Label
from haystack.preprocessor.utils import eval_data_from_file


logger = logging.getLogger(__name__)


Base = declarative_base()  # type: Any


class ORMBase(Base):
    __abstract__ = True

    id = Column(String(100), default=lambda: str(uuid4()), primary_key=True)
    created = Column(DateTime, server_default=func.now())
    updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now())


class DocumentORM(ORMBase):
    __tablename__ = "document"

    text = Column(Text, nullable=False)
    index = Column(String(100), nullable=False)
    vector_id = Column(String(100), unique=True, nullable=True)

    # speeds up queries for get_documents_by_vector_ids() by having a single query that returns joined metadata
    meta = relationship("MetaORM", backref="Document", lazy="joined")


class MetaORM(ORMBase):
    __tablename__ = "meta"

    name = Column(String(100), index=True)
    value = Column(String(1000), index=True)
    document_id = Column(String(100), ForeignKey("document.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)

    documents = relationship(DocumentORM, backref="Meta")


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
        update_existing_documents: bool = False,
    ):
        """
        :param url: URL for SQL database as expected by SQLAlchemy. More info here: https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
        :param index: The documents are scoped to an index attribute that can be used when writing, querying, or deleting documents. 
                      This parameter sets the default value for document index.
        :param label_index: The default value of index attribute for the labels.
        :param update_existing_documents: Whether to update any existing documents with the same ID when adding
                                          documents. When set as True, any document with an existing ID gets updated.
                                          If set to False, an error is raised if the document ID of the document being
                                          added already exists. Using this parameter coud cause performance degradation for document insertion. 
        """
        engine = create_engine(url)
        ORMBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.index = index
        self.label_index = label_index
        self.update_existing_documents = update_existing_documents
        if getattr(self, "similarity", None) is None:
            self.similarity = None

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        documents = self.get_documents_by_id([id], index)
        document = documents[0] if documents else None
        return document

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index
        results = self.session.query(DocumentORM).filter(DocumentORM.id.in_(ids), DocumentORM.index == index).all()
        documents = [self._convert_sql_row_to_document(row) for row in results]

        return documents

    def get_documents_by_vector_ids(self, vector_ids: List[str], index: Optional[str] = None):
        """Fetch documents by specifying a list of text vector id strings"""
        index = index or self.index
        results = self.session.query(DocumentORM).filter(
            DocumentORM.vector_id.in_(vector_ids),
            DocumentORM.index == index
        ).all()
        sorted_results = sorted(results, key=lambda doc: vector_ids.index(doc.vector_id))
        documents = [self._convert_sql_row_to_document(row) for row in sorted_results]
        return documents

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """

        index = index or self.index
        query = self.session.query(DocumentORM).filter_by(index=index)

        if filters:
            query = query.join(MetaORM)
            for key, values in filters.items():
                query = query.filter(MetaORM.name == key, MetaORM.value.in_(values))

        documents = [self._convert_sql_row_to_document(row) for row in query.all()]
        return documents

    def get_all_labels(self, index=None, filters: Optional[dict] = None):
        """
        Return all labels in the document store
        """
        index = index or self.label_index
        label_rows = self.session.query(LabelORM).filter_by(index=index).all()
        labels = [self._convert_sql_row_to_label(row) for row in label_rows]

        return labels

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        """
        Indexes documents for later queries.

      :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: add an optional index attribute to documents. It can be later used for filtering. For instance,
                      documents for evaluation can be indexed in a separate index than the documents for search.

        :return: None
        """

        # Make sure we comply to Document class format
        document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
        index = index or self.index
        for doc in document_objects:
            meta_fields = doc.meta or {}
            vector_id = meta_fields.get("vector_id")
            meta_orms = [MetaORM(name=key, value=value) for key, value in meta_fields.items()]
            doc_orm = DocumentORM(id=doc.id, text=doc.text, vector_id=vector_id, meta=meta_orms, index=index)
            if self.update_existing_documents:
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
        for label in labels:
            label_orm = LabelORM(
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
            self.session.add(label_orm)
        self.session.commit()

    def update_vector_ids(self, vector_id_map: Dict[str, str], index: Optional[str] = None):
        """
        Update vector_ids for given document_ids.

        :param vector_id_map: dict containing mapping of document_id -> vector_id.
        :param index: filter documents by the optional index attribute for documents in database.
        """
        index = index or self.index
        self.session.query(DocumentORM).filter(
            DocumentORM.id.in_(vector_id_map),
            DocumentORM.index == index
        ).update({
            DocumentORM.vector_id: case(
                vector_id_map,
                value=DocumentORM.id,
            )
        }, synchronize_session=False)
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

    def add_eval_data(self, filename: str, doc_index: str = "eval_document", label_index: str = "label"):
        """
        Adds a SQuAD-formatted file to the DocumentStore in order to be able to perform evaluation on it.

        :param filename: Name of the file containing evaluation data
        :type filename: str
        :param doc_index: Elasticsearch index where evaluation documents should be stored
        :type doc_index: str
        :param label_index: Elasticsearch index where labeled questions should be stored
        :type label_index: str
        """

        docs, labels = eval_data_from_file(filename)
        self.write_documents(docs, index=doc_index)
        self.write_labels(labels, index=label_index)

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
        index = index or self.index
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
        )
        return label

    def query_by_embedding(self,
                           query_emb: List[float],
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

        if filters:
            raise NotImplementedError("Delete by filters is not implemented for SQLDocumentStore.")
        index = index or self.index
        documents = self.session.query(DocumentORM).filter_by(index=index)
        documents.delete(synchronize_session=False)

    def _get_or_create(self, session, model, **kwargs):
        instance = session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance
        else:
            instance = model(**kwargs)
            session.add(instance)
            session.commit()
            return instance
