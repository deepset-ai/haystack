import uuid
from typing import Any, Dict, Union, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, ForeignKey, PickleType, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy_utils import UUIDType
from uuid import UUID
from haystack.indexing.utils import eval_data_from_file
from haystack.database.base import BaseDocumentStore, Document, Label

Base = declarative_base()  # type: Any


class ORMBase(Base):
    __abstract__ = True

    id = Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    created = Column(DateTime, server_default=func.now())
    updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now())


class DocumentORM(ORMBase):
    __tablename__ = "document"

    text = Column(String, nullable=False)
    index = Column(String, nullable=False)
    meta_data = Column(PickleType)

    tags = relationship("TagORM", secondary="document_tag", backref="Document")


class TagORM(ORMBase):
    __tablename__ = "tag"

    name = Column(String)
    value = Column(String)

    documents = relationship(DocumentORM, secondary="document_tag", backref="Tag")


class DocumentTagORM(ORMBase):
    __tablename__ = "document_tag"

    document_id = Column(UUIDType(binary=False), ForeignKey("document.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("tag.id"), nullable=False)


class LabelORM(ORMBase):
    __tablename__ = "label"

    document_id = Column(UUIDType(binary=False), ForeignKey("document.id"), nullable=False)
    index = Column(String, nullable=False)
    no_answer = Column(Boolean, nullable=False)
    origin = Column(String, nullable=False)
    question = Column(String, nullable=False)
    is_correct_answer = Column(Boolean, nullable=False)
    is_correct_document = Column(Boolean, nullable=False)
    answer = Column(String, nullable=False)
    offset_start_in_doc = Column(Integer, nullable=False)
    model_id = Column(Integer, nullable=True)


class SQLDocumentStore(BaseDocumentStore):
    def __init__(self, url: str = "sqlite://", index="document"):
        engine = create_engine(url)
        ORMBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.index = index
        self.label_index = "label"

    def get_document_by_id(self, id: UUID, index=None) -> Optional[Document]:
        index = index or self.index
        document_row = self.session.query(DocumentORM).filter_by(index=index, id=id).first()
        document = document_row or self._convert_sql_row_to_document(document_row)
        return document

    def get_all_documents(self, index=None) -> List[Document]:
        index = index or self.index
        document_rows = self.session.query(DocumentORM).filter_by(index=index).all()
        documents = [self._convert_sql_row_to_document(row) for row in document_rows]

        return documents

    def get_all_labels(self, index=None, filters: Optional[dict] = None):
        index = index or self.label_index
        label_rows = self.session.query(LabelORM).filter_by(index=index).all()
        labels = [self._convert_sql_row_to_label(row) for row in label_rows]

        return labels

    def get_document_ids_by_tags(self, tags: Dict[str, Union[str, List]], index: Optional[str] = None) -> List[str]:
        """
        Get list of document ids that have tags from the given list of tags.

        :param tags: limit scope to documents having the given tags and their corresponding values.
                     The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        """
        if not tags:
            raise Exception("No tag supplied for filtering the documents")

        if index:
            raise Exception("'index' parameter is not supported in SQLDocumentStore.get_document_ids_by_tags().")

        query = """
                  SELECT id FROM document WHERE id in (
                      SELECT dt.document_id
                      FROM document_tag dt JOIN
                          tag t
                          ON t.id = dt.tag_id
                      GROUP BY dt.document_id
              """
        tag_filters = []
        for tag in tags:
            tag_filters.append(f"SUM(CASE WHEN t.value='{tag}' THEN 1 ELSE 0 END) > 0")

        final_query = f"{query} HAVING {' AND '.join(tag_filters)});"
        query_results = self.session.execute(final_query)

        doc_ids = [row[0] for row in query_results]
        return doc_ids

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally, you can also supply "tags": ["one-tag", "another-one"]
                          or additional meta data via "meta": {"name": "<some-document-name>, "author": "someone", "url":"some-url" ...}
        :param index: add an optional index attribute to documents. It can be later used for filtering. For instance,
                      documents for evaluation can be indexed in a separate index than the documents for search.

        :return: None
        """

        # Make sure we comply to Document class format
        documents = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
        index = index or self.index
        for doc in documents:
            row = DocumentORM(id=doc.id, text=doc.text, meta_data=doc.meta, index=index)  # type: ignore
            self.session.add(row)
        self.session.commit()

    def write_labels(self, labels, index=None):

        labels = [Label.from_dict(l) if isinstance(l, dict) else l for l in labels]
        index = index or self.index
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

    def add_eval_data(self, filename: str, doc_index: str = "document", label_index: str = "label"):
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

    def get_document_count(self, index=None) -> int:
        index = index or self.index
        return self.session.query(DocumentORM).filter_by(index=index).count()

    def get_label_count(self, index: Optional[str] = None) -> int:
        index = index or self.index
        return self.session.query(LabelORM).filter_by(index=index).count()

    def _convert_sql_row_to_document(self, row) -> Document:
        document = Document(
            id=row.id,
            text=row.text,
            meta=row.meta_data,
            tags=row.tags
        )
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
                           index: Optional[str] = None) -> List[Document]:

        raise NotImplementedError("SQLDocumentStore is currently not supporting embedding queries. "
                                  "Change the query type (e.g. by choosing a different retriever) "
                                  "or change the DocumentStore (e.g. to ElasticsearchDocumentStore)")

    def delete_all_documents(self, index=None):
        """
        Delete all documents in a index.

        :param index: index name
        :return: None
        """

        index = index or self.index
        documents = self.session.query(DocumentORM).filter_by(index=index)
        documents.delete(synchronize_session=False)
