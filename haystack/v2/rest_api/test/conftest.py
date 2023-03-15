import pytest
from fastapi.testclient import TestClient

from haystack.v2.rest_api.app import get_app


@pytest.fixture(autouse=True)
def client():
    app = get_app()
    client = TestClient(app)
    return client
