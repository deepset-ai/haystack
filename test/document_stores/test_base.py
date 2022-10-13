import pytest
import numpy as np

from haystack.schema import Document, Label, Answer


@pytest.mark.document_store
class DocumentStoreTest:
    """
    This is the base class for any Document Store testsuite, it doesn't have the `Test` prefix in the name
    because we want to run its methods only in subclasses.
    """

    @pytest.fixture
    def documents(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "year": "2020", "month": "01"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "year": "2021", "month": "02"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"Document {i} without embeddings",
                    meta={"name": f"name_{i}", "no_embedding": True, "month": "03"},
                )
            )

        return documents

    @pytest.fixture
    def labels(self, documents):
        labels = []
        for i, d in enumerate(documents):
            labels.append(
                Label(
                    query="query",
                    document=d,
                    is_correct_document=True,
                    is_correct_answer=False,
                    # create a mix set of labels
                    origin="user-feedback" if i % 2 else "gold-label",
                    answer=None if not i else Answer(f"the answer is {i}"),
                )
            )
        return labels

    @pytest.mark.integration
    def test_write_documents(self, ds, documents):
        ds.write_documents(documents)
        docs = ds.get_all_documents()
        assert len(docs) == len(documents)
        for i, doc in enumerate(docs):
            expected = documents[i]
            assert doc.id == expected.id

    @pytest.mark.integration
    def test_write_labels(self, ds, labels):
        ds.write_labels(labels)
        assert ds.get_all_labels() == labels
