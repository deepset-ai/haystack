import pytest

from haystack.preview import Document
from haystack.preview.components.batch.batch_processor import BatchProcessor, BatchProcessorError


class TestBatchCreator:
    @pytest.mark.unit
    def test_to_dict_builtin_type(self):
        component = BatchProcessor(expected_type=int)
        data = component.to_dict()
        assert data == {"type": "BatchProcessor", "init_parameters": {"expected_type": "int"}}

    @pytest.mark.unit
    def test_to_dict_object_type(self):
        component = BatchProcessor(expected_type=Document)
        data = component.to_dict()
        assert data == {
            "type": "BatchProcessor",
            "init_parameters": {"expected_type": "haystack.preview.dataclasses.document.Document"},
        }

    @pytest.mark.unit
    def test_from_dict_builtin_type(self):
        data = {"type": "BatchProcessor", "init_parameters": {"expected_type": "int"}}
        component = BatchProcessor.from_dict(data)
        assert component.expected_type == int

    @pytest.mark.unit
    def test_from_dict_object_type(self):
        data = {
            "type": "BatchProcessor",
            "init_parameters": {"expected_type": "haystack.preview.dataclasses.document.Document"},
        }
        component = BatchProcessor.from_dict(data)
        assert component.expected_type == Document

    @pytest.mark.unit
    def test_run_with_no_input(self):
        component = BatchProcessor(expected_type=int)
        output = component.run()
        assert output == {"item": None, "current_batch": None}

    @pytest.mark.unit
    def test_run_with_new_batch(self):
        component = BatchProcessor(expected_type=int)
        output = component.run(new_batch=[1, 2, 3])
        assert output == {"item": 1, "current_batch": [2, 3]}

    @pytest.mark.unit
    def test_run_with_current_batch(self):
        component = BatchProcessor(expected_type=int)
        output = component.run(current_batch=[1, 2, 3])
        assert output == {"item": 1, "current_batch": [2, 3]}

    @pytest.mark.unit
    def test_run_with_new_and_current_batch(self):
        component = BatchProcessor(expected_type=int)
        with pytest.raises(
            BatchProcessorError,
            match="BatchProcessor received a new batch before the previous one was fully processed.",
        ):
            component.run(new_batch=[1, 2, 3], current_batch=[1, 2, 3])
