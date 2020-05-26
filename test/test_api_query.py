import pytest
from fastapi.testclient import TestClient


def test_example_model(monkeypatch):
    with monkeypatch.context() as m:
        with pytest.raises(AttributeError) as _:
            m.delattr("haystack.database.elasticsearch")
            from haystack.api.application import app
            client = TestClient(app)
            client.post("/models/1/doc-qa", json={"questions": ["Who is the father of George Orwell?"]})
