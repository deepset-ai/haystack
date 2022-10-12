import pytest


class TestDocumentStore:
    @pytest.mark.integration
    def test_write_documents(self, ds, documents):
        ds.write_documents(documents)
        docs = ds.get_all_documents()
        assert len(docs) == len(documents)
        for i, doc in enumerate(docs):
            expected = documents[i]
            assert doc.id == expected.id
