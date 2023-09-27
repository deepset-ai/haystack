import logging

import pytest

from haystack.preview.components.file_converters.pypdf import PyPDFToDocument


class TestPyPDFToDocument:
    @pytest.mark.unit
    def test_to_dict(self):
        component = PyPDFToDocument()
        data = component.to_dict()
        assert data == {"type": "PyPDFToDocument", "init_parameters": {"id_hash_keys": []}}

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = PyPDFToDocument(id_hash_keys=["name"])
        data = component.to_dict()
        assert data == {"type": "PyPDFToDocument", "init_parameters": {"id_hash_keys": ["name"]}}

    @pytest.mark.unit
    def test_from_dict(self):
        data = {"type": "PyPDFToDocument", "init_parameters": {"id_hash_keys": ["name"]}}
        component = PyPDFToDocument.from_dict(data)
        assert component.id_hash_keys == ["name"]

    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        """
        Test if the component runs correctly.
        """
        paths = [preview_samples_path / "pdf" / "react_paper.pdf"]
        converter = PyPDFToDocument()
        output = converter.run(paths=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "ReAct" in docs[0].text

    @pytest.mark.unit
    def test_run_error_handling(self, preview_samples_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.pdf"]
        converter = PyPDFToDocument()
        with caplog.at_level(logging.WARNING):
            converter.run(paths=paths)
            assert "Could not read file non_existing_file.pdf" in caplog.text
