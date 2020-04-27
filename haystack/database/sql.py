from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from haystack.database.base import BaseDocumentStore, Document as DocumentSchema

Base = declarative_base()


class ORMBase(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, server_default=func.now())
    updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now())


class Document(ORMBase):
    __tablename__ = "document"

    name = Column(String)
    text = Column(String)

    tags = relationship("Tag", secondary="document_tag", backref="Document")


class Tag(ORMBase):
    __tablename__ = "tag"

    name = Column(String)
    value = Column(String)

    documents = relationship("Document", secondary="document_tag", backref="Tag")


class DocumentTag(ORMBase):
    __tablename__ = "document_tag"

    document_id = Column(Integer, ForeignKey("document.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("tag.id"), nullable=False)


class SQLDocumentStore(BaseDocumentStore):
    def __init__(self, url="sqlite://"):
        engine = create_engine(url)
        ORMBase.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_document_by_id(self, id):
        document_row = self.session.query(Document).get(id)
        document = self._convert_sql_row_to_document(document_row)

        return document

    def get_all_documents(self):
        document_rows = self.session.query(Document).all()
        documents = []
        for row in document_rows:
            documents.append(self._convert_sql_row_to_document(row))

        return documents

    def get_document_ids_by_tags(self, tags):
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

    def write_documents(self, documents):
        for doc in documents:
            row = Document(name=doc["name"], text=doc["text"])
            self.session.add(row)
        self.session.commit()

    def get_document_count(self):
        return self.session.query(Document).count()

    def _convert_sql_row_to_document(self, row) -> Document:
        document = DocumentSchema(
            id=row.id,
            text=row.text,
            meta=row.tags
        )
        return document
