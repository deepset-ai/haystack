from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.database.orm import Document


def test_db_write_read():
    from haystack.database import db
    db.drop_all()
    db.create_all()

    write_documents_to_db(document_dir="samples/docs")
    documents = db.session.query(Document).order_by(Document.text).all()
    assert len(documents) == 2
    assert documents[0].text == 'A Doc specifically talking about haystack.\nHaystack can be used to scale QA models to large document collections.'
