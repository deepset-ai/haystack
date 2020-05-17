from fastapi.testclient import TestClient

from haystack.api.application import app

client = TestClient(app)


def test_root_404():
    response = client.get("/")
    assert response.status_code == 404


def test_example_model():
    import pytest
    with pytest.raises(AttributeError) as _:
        client.post("/models/1/doc-qa", json={"questions": ["Who is the father of George Orwell?"]})
