# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

import pytest

from haystack.components.converters import HTMLToDocument
from haystack.dataclasses import ByteStream


class TestHTMLToDocument:
    def test_run(self, test_files_path):
        """
        Test if the component runs correctly.
        """
        sources = [test_files_path / "html" / "what_is_haystack.html"]
        converter = HTMLToDocument()
        results = converter.run(sources=sources, meta={"test": "TEST"})
        docs = results["documents"]
        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta["test"] == "TEST"

    def test_run_different_extractors(self, test_files_path):
        """
        Test if the component runs correctly with different boilrepy3 extractors.
        """
        sources = [test_files_path / "html" / "what_is_haystack.html"]

        converter_article = HTMLToDocument(extractor_type="ArticleExtractor")
        converter_keep_everything = HTMLToDocument(extractor_type="KeepEverythingExtractor")

        doc_article = converter_article.run(sources=sources)["documents"][0]
        doc_keep_everything = converter_keep_everything.run(sources=sources)["documents"][0]

        assert len(doc_keep_everything.content) > len(doc_article.content)

    def test_run_doc_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is supplied by the user.
        """
        converter = HTMLToDocument()
        sources = [test_files_path / "html" / "what_is_haystack.html"]
        metadata = [{"file_name": "what_is_haystack.html"}]
        results = converter.run(sources=sources, meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta["file_name"] == "what_is_haystack.html"

    def test_incorrect_meta(self, test_files_path):
        """
        Test if the component raises an error when incorrect metadata is supplied by the user.
        """
        converter = HTMLToDocument()
        sources = [test_files_path / "html" / "what_is_haystack.html"]
        metadata = [{"file_name": "what_is_haystack.html"}, {"file_name": "haystack.html"}]
        with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
            converter.run(sources=sources, meta=metadata)

    def test_run_bytestream_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object.
        """
        converter = HTMLToDocument()
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            byte_stream = file.read()
            stream = ByteStream(byte_stream, meta={"content_type": "text/html", "url": "test_url"})

        results = converter.run(sources=[stream])
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta == {"content_type": "text/html", "url": "test_url"}

    def test_run_bytestream_and_doc_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object and supplied by the user.

        There is no overlap between the metadata received.
        """
        converter = HTMLToDocument()
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            byte_stream = file.read()
            stream = ByteStream(byte_stream, meta={"content_type": "text/html", "url": "test_url"})

        metadata = [{"file_name": "what_is_haystack.html"}]
        results = converter.run(sources=[stream], meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta == {"file_name": "what_is_haystack.html", "content_type": "text/html", "url": "test_url"}

    def test_run_bytestream_doc_overlapping_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object and supplied by the user.

        There is an overlap between the metadata received.

        The component should use the supplied metadata to overwrite the values if there is an overlap between the keys.
        """
        converter = HTMLToDocument()
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            byte_stream = file.read()
            # ByteStream has "url" present in metadata
            stream = ByteStream(byte_stream, meta={"content_type": "text/html", "url": "test_url_correct"})

        # "url" supplied by the user overwrites value present in metadata
        metadata = [{"file_name": "what_is_haystack.html", "url": "test_url_new"}]
        results = converter.run(sources=[stream], meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta == {
            "file_name": "what_is_haystack.html",
            "content_type": "text/html",
            "url": "test_url_new",
        }

    def test_run_wrong_file_type(self, test_files_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        sources = [test_files_path / "audio" / "answer.wav"]
        converter = HTMLToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Failed to extract text from" in caplog.text

        assert results["documents"] == []

    def test_run_error_handling(self, caplog):
        """
        Test if the component correctly handles errors.
        """
        sources = ["non_existing_file.html"]
        converter = HTMLToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Could not read non_existing_file.html" in caplog.text
            assert results["documents"] == []

    def test_mixed_sources_run(self, test_files_path):
        """
        Test if the component runs correctly if the input is a mix of paths and ByteStreams.
        """
        sources = [
            test_files_path / "html" / "what_is_haystack.html",
            str((test_files_path / "html" / "what_is_haystack.html").absolute()),
        ]
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as f:
            byte_stream = f.read()
            sources.append(ByteStream(byte_stream))

        converter = HTMLToDocument()
        results = converter.run(sources=sources)
        docs = results["documents"]
        assert len(docs) == 3
        for doc in docs:
            assert "Haystack" in doc.content

    def test_serde(self):
        """
        Test if the component runs correctly gets serialized and deserialized.
        """
        converter = HTMLToDocument("ArticleExtractor")
        serde_data = converter.to_dict()
        new_converter = HTMLToDocument.from_dict(serde_data)
        assert new_converter.extractor_type == converter.extractor_type
        assert new_converter.try_others == converter.try_others

    def test_run_try_others_false(self, test_files_path, caplog):
        converter = HTMLToDocument(try_others=False)
        result = converter.run(sources=[Path(test_files_path / "html" / "paul_graham_superlinear.html")])

        # paul_graham_superlinear.html is a page that the DefaultExtractor cannot extract text from
        assert len(result["documents"]) == 0
        assert "Failed to extract text from" in caplog.text
        assert "Skipping it" in caplog.text

    def test_run_try_others_true(self, test_files_path, caplog):
        # try_others=True is the default value
        converter = HTMLToDocument()
        result = converter.run(sources=[Path(test_files_path / "html" / "paul_graham_superlinear.html")])

        # paul_graham_superlinear.html is a page that the DefaultExtractor cannot extract text from
        assert len(result["documents"]) == 1
        assert "Superlinear" in result["documents"][0].content
