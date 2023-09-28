import logging

import pytest

from haystack.preview.components.file_converters import HTMLToDocument


class TestHTMLToDocument:
    @pytest.mark.unit
    def test_to_dict(self):
        component = HTMLToDocument()
        data = component.to_dict()
        assert data == {"type": "HTMLToDocument", "init_parameters": {"id_hash_keys": []}}

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = HTMLToDocument(id_hash_keys=["name"])
        data = component.to_dict()
        assert data == {"type": "HTMLToDocument", "init_parameters": {"id_hash_keys": ["name"]}}

    @pytest.mark.unit
    def test_from_dict(self):
        data = {"type": "HTMLToDocument", "init_parameters": {"id_hash_keys": ["name"]}}
        component = HTMLToDocument.from_dict(data)
        assert component.id_hash_keys == ["name"]

    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        """
        Test if the component runs correctly.
        """
        paths = [preview_samples_path / "html" / "what_is_haystack.html"]
        converter = HTMLToDocument()
        output = converter.run(paths=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "Haystack" in docs[0].text

    @pytest.mark.unit
    def test_run_wrong_file_type(self, preview_samples_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        paths = [preview_samples_path / "audio" / "answer.wav"]
        converter = HTMLToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(paths=paths)
            assert "codec can't decode byte" in caplog.text

        docs = output["documents"]
        assert docs == []

    @pytest.mark.unit
    def test_run_error_handling(self, preview_samples_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.html"]
        converter = HTMLToDocument()
        with caplog.at_level(logging.WARNING):
            result = converter.run(paths=paths)
            assert "Could not read file non_existing_file.html" in caplog.text
            assert result["documents"] == []
