import uuid
from typing import Any, Dict, Union, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, ForeignKey, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy_utils import UUIDType
from uuid import UUID
from haystack.database.base import BaseDocumentStore, Document

Base = declarative_base()  # type: Any


class ORMBase(Base):
    __abstract__ = True

    id = Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    created = Column(DateTime, server_default=func.now())
    updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now())


class DocumentORM(ORMBase):
    __tablename__ = "document"

    text = Column(String)
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


class SQLDocumentStore(BaseDocumentStore):
    def __init__(self, url: str = "sqlite://"):
        engine = create_engine(url)
        ORMBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_document_by_id(self, id: UUID) -> Optional[Document]:
        document_row = self.session.query(DocumentORM).get(id)
        document = self._convert_sql_row_to_document(document_row)

        return document

    def get_all_documents(self) -> List[Document]:
        document_rows = self.session.query(DocumentORM).all()
        documents = []
        for row in document_rows:
            documents.append(self._convert_sql_row_to_document(row))

        return documents

    def get_document_ids_by_tags(self, tags: Dict[str, Union[str, List]]) -> List[str]:
        """
        Get list of document ids that have tags from the given list of tags.

        :param tags: limit scope to documents having the given tags and their corresponding values.
                     The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        """
        if not tags:
            raise Exception("No tag supplied for filtering the documents")

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

    def write_documents(self, documents: Union[List[dict], List[Document]]):
        """
        Indexes documents for later queries.

        :param documents: List of dictionaries in the format {"text": "<the-actual-text>"}.
                          Optionally, you can also supply meta data via "meta": {"author": "someone", "url":"some-url" ...}

        :return: None
        """

        # Make sure we comply to Document class format
        documents = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]

        for doc in documents:
            row = DocumentORM(text=doc.text, meta_data=doc.meta)
            self.session.add(row)
        self.session.commit()

    def get_document_count(self) -> int:
        return self.session.query(DocumentORM).count()

    def _convert_sql_row_to_document(self, row) -> Document:
        document = Document(
            id=row.id,
            text=row.text,
            meta=row.meta_data,
            tags=row.tags
        )
        return document

    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None) -> List[Document]:

        raise NotImplementedError("SQLDocumentStore is currently not supporting embedding queries. "
                                  "Change the query type (e.g. by choosing a different retriever) "
                                  "or change the DocumentStore (e.g. to ElasticsearchDocumentStore)")
