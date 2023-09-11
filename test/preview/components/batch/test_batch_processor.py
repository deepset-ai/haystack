from typing import Any
from unittest.mock import MagicMock

import pytest

from haystack.preview import Document, DeserializationError
from haystack.preview.testing.factory import document_store_class
from haystack.preview.components.batch.batch_creator import BatchCreator
from haystack.preview.document_stores import DuplicatePolicy


class TestBatchCreator:
    @pytest.mark.unit
    def test_to_dict_builtin_type(self):
        component = BatchCreator(expected_type=int, max_batch_size=10)
        data = component.to_dict()
        assert data == {"type": "BatchCreator", "init_parameters": {"expected_type": "int", "max_batch_size": 10}}

    @pytest.mark.unit
    def test_to_dict_object_type(self):
        component = BatchCreator(expected_type=Document, max_batch_size=10)
        data = component.to_dict()
        assert data == {
            "type": "BatchCreator",
            "init_parameters": {
                "expected_type": "haystack.preview.dataclasses.document.Document",
                "max_batch_size": 10,
            },
        }

    @pytest.mark.unit
    def test_from_dict_builtin_type(self):
        data = {"type": "BatchCreator", "init_parameters": {"expected_type": "int", "max_batch_size": 10}}
        component = BatchCreator.from_dict(data)
        assert component.expected_type == int
        assert component.max_batch_size == 10

    @pytest.mark.unit
    def test_from_dict_object_type(self):
        data = {
            "type": "BatchCreator",
            "init_parameters": {
                "expected_type": "haystack.preview.dataclasses.document.Document",
                "max_batch_size": 10,
            },
        }
        component = BatchCreator.from_dict(data)
        assert component.expected_type == Document
        assert component.max_batch_size == 10

    @pytest.mark.unit
    def test_run_default(self):
        component = BatchCreator(expected_type=int, max_batch_size=3)
        assert component.batch == []

        output = component.run(item=1)
        assert output == {}
        assert component.batch == [1]

        output = component.run(item=2)
        assert output == {}
        assert component.batch == [1, 2]

        output = component.run(item=3)
        assert output == {"batch": [1, 2, 3]}
        assert component.batch == []

    @pytest.mark.unit
    def test_run_with_release_batch(self):
        component = BatchCreator(expected_type=int, max_batch_size=10)
        assert component.batch == []

        output = component.run(item=1)
        assert output == {}
        assert component.batch == [1]

        output = component.run(item=2, release_batch=True)
        assert output == {"batch": [1, 2]}
        assert component.batch == []
