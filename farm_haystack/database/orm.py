from sqlalchemy.orm import relationship

from farm_haystack.database import db


class ORMBase(db.Model):
    __abstract__ = True

    id = db.Column(db.Integer, primary_key=True)
    created = db.Column(db.DateTime, server_default=db.func.now())
    updated = db.Column(
        db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now()
    )


class Document(ORMBase):
    name = db.Column(db.String)
    text = db.Column(db.String)

    tags = relationship("Tag", secondary="document_tag", backref="Document")


class Tag(ORMBase):
    name = db.Column(db.String)
    value = db.Column(db.String)

    documents = relationship("Document", secondary="document_tag", backref="Tag")


class DocumentTag(ORMBase):
    document_id = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=False)
    tag_id = db.Column(db.Integer, db.ForeignKey("tag.id"), nullable=False)
