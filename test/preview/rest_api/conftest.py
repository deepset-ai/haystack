from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import haystack.preview.rest_api.app as app_module


@pytest.fixture(autouse=True)
def client(monkeypatch):
    monkeypatch.setattr(app_module, "DEFAULT_PIPELINES", Path(__file__).parent / "pipelines" / "default.json")
    app = app_module.get_app()
    client = TestClient(app)
    return client
