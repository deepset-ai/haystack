# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import mimetypes
import sys
from pathlib import PosixPath
from unittest.mock import mock_open, patch

import pytest
from packaging import version

import haystack
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.routers.file_type_router import FileTypeRouter
from haystack.dataclasses import ByteStream


@pytest.mark.skipif(
    sys.platform in ["win32", "cygwin"],
    reason="Can't run on Windows Github CI, need access to registry to get mime types",
)
class TestFileTypeRouter:
    def test_init(self):
        """
        Test that the component initializes correctly.
        """
        router = FileTypeRouter(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        assert router.mime_types == ["text/plain", "audio/x-wav", "image/jpeg"]
        assert router._additional_mimetypes is None

        router = FileTypeRouter(
            mime_types=["text/plain"],
            additional_mimetypes={"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"},
        )
        assert router.mime_types == ["text/plain"]
        assert router._additional_mimetypes == {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"
        }

    def test_init_fail_wo_mime_types(self):
        """
        Test that the component raises an error if no mime types are provided.
        """
        with pytest.raises(ValueError):
            FileTypeRouter(mime_types=[])

    def test_to_dict(self):
        router = FileTypeRouter(
            mime_types=["text/plain", "audio/x-wav", "image/jpeg"],
            additional_mimetypes={"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"},
        )
        expected_dict = {
            "type": "haystack.components.routers.file_type_router.FileTypeRouter",
            "init_parameters": {
                "mime_types": ["text/plain", "audio/x-wav", "image/jpeg"],
                "additional_mimetypes": {
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"
                },
                "raise_on_failure": False,
            },
        }
        assert router.to_dict() == expected_dict

    def test_from_dict(self):
        router_dict = {
            "type": "haystack.components.routers.file_type_router.FileTypeRouter",
            "init_parameters": {
                "mime_types": ["text/plain", "audio/x-wav", "image/jpeg"],
                "additional_mimetypes": {
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"
                },
            },
        }
        loaded_router = FileTypeRouter.from_dict(router_dict)

        expected_router = FileTypeRouter(
            mime_types=["text/plain", "audio/x-wav", "image/jpeg"],
            additional_mimetypes={"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"},
        )

        assert loaded_router.mime_types == expected_router.mime_types
        assert loaded_router._additional_mimetypes == expected_router._additional_mimetypes

    def test_run(self, test_files_path):
        """
        Test if the component runs correctly in the simplest happy path.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "txt" / "doc_2.txt",
            test_files_path / "audio" / "the context for this answer is here.wav",
            test_files_path / "images" / "apple.jpg",
            test_files_path / "msg" / "sample.msg",
        ]

        router = FileTypeRouter(mime_types=[r"text/plain", r"audio/x-wav", r"image/jpeg", "application/vnd.ms-outlook"])
        output = router.run(sources=file_paths)
        assert output
        assert len(output[r"text/plain"]) == 2
        assert len(output[r"audio/x-wav"]) == 1
        assert len(output[r"image/jpeg"]) == 1
        assert len(output["application/vnd.ms-outlook"]) == 1
        assert not output.get("unclassified")

    def test_run_with_single_meta(self, test_files_path):
        """
        Test if the component runs correctly when a single metadata dictionary is provided.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "txt" / "doc_2.txt",
            test_files_path / "audio" / "the context for this answer is here.wav",
        ]

        meta = {"meta_field": "meta_value"}

        router = FileTypeRouter(mime_types=[r"text/plain", r"audio/x-wav"])
        output = router.run(sources=file_paths, meta=meta)
        assert output

        assert len(output[r"text/plain"]) == 2
        assert len(output[r"audio/x-wav"]) == 1
        assert not output.get("unclassified")

        for elements in output.values():
            for el in elements:
                assert isinstance(el, ByteStream)
                assert el.meta["meta_field"] == "meta_value"

    def test_run_with_meta_list(self, test_files_path):
        """
        Test if the component runs correctly when a list of metadata dictionaries is provided.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "images" / "apple.jpg",
            test_files_path / "audio" / "the context for this answer is here.wav",
        ]

        meta = [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}]

        router = FileTypeRouter(mime_types=[r"text/plain", r"audio/x-wav", r"image/jpeg"])
        output = router.run(sources=file_paths, meta=meta)
        assert output

        assert len(output[r"text/plain"]) == 1
        assert len(output[r"audio/x-wav"]) == 1
        assert len(output[r"image/jpeg"]) == 1
        assert not output.get("unclassified")

        for i, elements in enumerate(output.values()):
            for el in elements:
                assert isinstance(el, ByteStream)

                expected_meta_key, expected_meta_value = list(meta[i].items())[0]
                assert el.meta[expected_meta_key] == expected_meta_value

    def test_run_with_meta_and_bytestreams(self):
        """
        Test if the component runs correctly with ByteStream inputs and meta.
        The original meta is preserved and the new meta is added.
        """

        bs = ByteStream.from_string("Haystack!", mime_type="text/plain", meta={"foo": "bar"})

        meta = {"another_key": "another_value"}

        router = FileTypeRouter(mime_types=[r"text/plain"])

        output = router.run(sources=[bs], meta=meta)

        assert output
        assert len(output[r"text/plain"]) == 1
        assert not output.get("unclassified")

        assert isinstance(output[r"text/plain"][0], ByteStream)
        assert output[r"text/plain"][0].meta["foo"] == "bar"
        assert output[r"text/plain"][0].meta["another_key"] == "another_value"

    def test_run_fails_if_meta_length_does_not_match_sources(self, test_files_path):
        """
        Test that the component raises an error if the length of the metadata list does not match the number of sources.
        """
        file_paths = [test_files_path / "txt" / "doc_1.txt"]

        meta = [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}]

        router = FileTypeRouter(mime_types=[r"text/plain"])

        with pytest.raises(ValueError):
            router.run(sources=file_paths, meta=meta)

    def test_run_with_bytestreams(self, test_files_path):
        """
        Test if the component runs correctly with ByteStream inputs.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "txt" / "doc_2.txt",
            test_files_path / "audio" / "the context for this answer is here.wav",
            test_files_path / "images" / "apple.jpg",
        ]
        mime_types = [r"text/plain", r"text/plain", r"audio/x-wav", r"image/jpeg"]
        # Convert file paths to ByteStream objects and set metadata
        byte_streams = []
        for path, mime_type in zip(file_paths, mime_types):
            stream = ByteStream(path.read_bytes())
            stream.mime_type = mime_type
            byte_streams.append(stream)

        # add unclassified ByteStream
        bs = ByteStream(b"unclassified content")
        bs.meta["content_type"] = "unknown_type"
        byte_streams.append(bs)

        router = FileTypeRouter(mime_types=[r"text/plain", r"audio/x-wav", r"image/jpeg"])
        output = router.run(sources=byte_streams)
        assert output
        assert len(output[r"text/plain"]) == 2
        assert len(output[r"audio/x-wav"]) == 1
        assert len(output[r"image/jpeg"]) == 1
        assert len(output.get("unclassified")) == 1

    def test_run_with_bytestreams_and_file_paths(self, test_files_path):
        """
        Test if the component raises an error for unsupported data source types.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "audio" / "the context for this answer is here.wav",
            test_files_path / "txt" / "doc_2.txt",
            test_files_path / "images" / "apple.jpg",
            test_files_path / "markdown" / "sample.md",
        ]
        mime_types = [r"text/plain", r"audio/x-wav", r"text/plain", r"image/jpeg", r"text/markdown"]
        byte_stream_sources = []
        for path, mime_type in zip(file_paths, mime_types):
            stream = ByteStream(path.read_bytes())
            stream.mime_type = mime_type
            byte_stream_sources.append(stream)

        mixed_sources = file_paths[:2] + byte_stream_sources[2:]

        router = FileTypeRouter(mime_types=[r"text/plain", r"audio/x-wav", r"image/jpeg", r"text/markdown"])
        output = router.run(sources=mixed_sources)
        assert len(output[r"text/plain"]) == 2
        assert len(output[r"audio/x-wav"]) == 1
        assert len(output[r"image/jpeg"]) == 1
        assert len(output[r"text/markdown"]) == 1

    def test_no_files(self):
        """
        Test that the component runs correctly when no files are provided.
        """
        router = FileTypeRouter(mime_types=[r"text/plain", r"audio/x-wav", r"image/jpeg"])
        output = router.run(sources=[])
        assert not output

    def test_unlisted_extensions(self, test_files_path):
        """
        Test that the component correctly handles files with non specified mime types.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "audio" / "ignored.mp3",
            test_files_path / "audio" / "this is the content of the document.wav",
        ]
        router = FileTypeRouter(mime_types=[r"text/plain"])
        output = router.run(sources=file_paths)
        assert len(output[r"text/plain"]) == 1
        assert "mp3" not in output
        assert len(output.get("unclassified")) == 2

    def test_no_extension(self, test_files_path):
        """
        Test that the component ignores files with no extension.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "txt" / "doc_2",
            test_files_path / "txt" / "doc_2.txt",
        ]
        router = FileTypeRouter(mime_types=[r"text/plain"])
        output = router.run(sources=file_paths)
        assert len(output[r"text/plain"]) == 2
        assert len(output.get("unclassified")) == 1

    def test_unsupported_source_type(self):
        """
        Test if the component raises an error for unsupported data source types.
        """
        router = FileTypeRouter(mime_types=[r"text/plain", r"audio/x-wav", r"image/jpeg"])
        with pytest.raises(ValueError, match="Unsupported data source type:"):
            router.run(sources=[{"unsupported": "type"}])

    def test_invalid_regex_pattern(self):
        """
        Test that the component raises a ValueError for invalid regex patterns.
        """
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            FileTypeRouter(mime_types=["[Invalid-Regex"])

    def test_regex_mime_type_matching(self, test_files_path):
        """
        Test if the component correctly matches mime types using regex.
        """
        router = FileTypeRouter(mime_types=[r"text\/.*", r"audio\/.*", r"image\/.*"])
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "audio" / "the context for this answer is here.wav",
            test_files_path / "images" / "apple.jpg",
        ]
        output = router.run(sources=file_paths)
        assert len(output[r"text\/.*"]) == 1, "Failed to match text file with regex"
        assert len(output[r"audio\/.*"]) == 1, "Failed to match audio file with regex"
        assert len(output[r"image\/.*"]) == 1, "Failed to match image file with regex"

    @patch("pathlib.Path.open", new_callable=mock_open, read_data=b"Mock file content.")
    def test_exact_mime_type_matching(self, mock_file):
        """
        Test if the component correctly matches mime types exactly, without regex patterns.
        """
        txt_stream = ByteStream(io.BytesIO(b"Text file content").read())
        txt_stream.mime_type = "text/plain"
        jpg_stream = ByteStream(io.BytesIO(b"JPEG file content").read())
        jpg_stream.mime_type = "image/jpeg"
        mp3_stream = ByteStream(io.BytesIO(b"MP3 file content").read())
        mp3_stream.mime_type = "audio/mpeg"

        byte_streams = [txt_stream, jpg_stream, mp3_stream]

        router = FileTypeRouter(mime_types=["text/plain", "image/jpeg"])

        output = router.run(sources=byte_streams)

        assert len(output["text/plain"]) == 1, "Failed to match 'text/plain' MIME type exactly"
        assert txt_stream in output["text/plain"], "'doc_1.txt' ByteStream not correctly classified as 'text/plain'"

        assert len(output["image/jpeg"]) == 1, "Failed to match 'image/jpeg' MIME type exactly"
        assert jpg_stream in output["image/jpeg"], "'apple.jpg' ByteStream not correctly classified as 'image/jpeg'"

        assert len(output.get("unclassified")) == 1, "Failed to handle unclassified file types"
        assert mp3_stream in output["unclassified"], "'sound.mp3' ByteStream should be unclassified but is not"

    def test_serde_in_pipeline(self):
        """
        Test if a pipeline containing the component can be serialized and deserialized without errors.
        """

        file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf"])

        pipeline = Pipeline()
        pipeline.add_component(instance=file_type_router, name="file_type_router")

        pipeline_dict = pipeline.to_dict()

        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "file_type_router": {
                    "type": "haystack.components.routers.file_type_router.FileTypeRouter",
                    "init_parameters": {
                        "mime_types": ["text/plain", "application/pdf"],
                        "additional_mimetypes": None,
                        "raise_on_failure": False,
                    },
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    @pytest.mark.integration
    def test_pipeline_with_converters(self, test_files_path):
        """
        Test if the component runs correctly in a pipeline with converters and passes metadata correctly.
        """
        file_type_router = FileTypeRouter(
            mime_types=["text/plain", "application/pdf"],
            additional_mimetypes={"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"},
        )

        pipe = Pipeline()
        pipe.add_component(instance=file_type_router, name="file_type_router")
        pipe.add_component(instance=TextFileToDocument(), name="text_file_converter")
        pipe.add_component(instance=PyPDFToDocument(), name="pypdf_converter")
        pipe.connect("file_type_router.text/plain", "text_file_converter.sources")
        pipe.connect("file_type_router.application/pdf", "pypdf_converter.sources")

        file_paths = [test_files_path / "txt" / "doc_1.txt", test_files_path / "pdf" / "sample_pdf_1.pdf"]

        meta = [{"meta_field_1": "meta_value_1"}, {"meta_field_2": "meta_value_2"}]

        output = pipe.run(data={"file_type_router": {"sources": file_paths, "meta": meta}})

        assert output["text_file_converter"]["documents"][0].meta["meta_field_1"] == "meta_value_1"
        assert output["pypdf_converter"]["documents"][0].meta["meta_field_2"] == "meta_value_2"

    def test_additional_mimetypes_integration(self, tmp_path):
        """
        Test if the component runs correctly in a pipeline with additional mimetypes correctly.
        """
        custom_mime_type = "application/x-spam"
        custom_extension = ".spam"
        test_file = tmp_path / f"test.{custom_extension}"
        test_file.touch()

        # confirm that mimetypes module doesn't know about this extension by default
        assert custom_mime_type not in mimetypes.types_map.values()

        # make haystack aware of the custom mime type
        router = FileTypeRouter(
            mime_types=[custom_mime_type], additional_mimetypes={custom_mime_type: custom_extension}
        )
        mappings = router.run(sources=[test_file])

        # assert the file was classified under the custom mime type
        assert custom_mime_type in mappings
        assert test_file in mappings[custom_mime_type]

    @pytest.mark.skipif(
        version.parse(haystack.__version__) >= version.parse("2.17.0"),
        reason="https://github.com/deepset-ai/haystack/pull/9573#issuecomment-3045237341",
    )
    def test_non_existent_file(self):
        """
        Test conditional FileNotFoundError behavior in FileTypeRouter.

        In Haystack versions prior to 2.17.0, `FileTypeRouter` does not raise an error
        when a non-existent file is passed without `meta`. However, it raises a
        FileNotFoundError when the same file is passed with `meta` supplied.

        This inconsistent behavior is slated to change in 2.17.0.
        See: https://github.com/deepset-ai/haystack/pull/9573#issuecomment-3045237341
        """
        router = FileTypeRouter(mime_types=[r"text/plain"])

        # No meta - does not raise error
        result = router.run(sources=["non_existent.txt"])
        assert result == {"text/plain": [PosixPath("non_existent.txt")]}

        # With meta - raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            router.run(sources=["non_existent.txt"], meta={"spam": "eggs"})

    def test_warning_for_non_existent_file(self):
        """
        Test that a warning is triggered when a non-existent file is encountered
        and raise_on_failure is False.
        """
        import warnings

        router = FileTypeRouter(mime_types=[r"text/plain"], raise_on_failure=False)

        # Capture warnings to verify they are triggered
        with pytest.warns(UserWarning, match="File not found: .* - skipping."):
            result = router.run(sources=["non_existent_file.txt"])

        # Verify the file is still included in the result despite the warning
        assert "text/plain" in result
        assert PosixPath("non_existent_file.txt") in result["text/plain"]

    def test_no_warning_when_raise_on_failure_is_true(self):
        """
        Test that no warning is triggered when raise_on_failure is True,
        instead a FileNotFoundError is raised.
        """
        router = FileTypeRouter(mime_types=[r"text/plain"], raise_on_failure=True)

        # Should raise FileNotFoundError instead of warning
        with pytest.raises(FileNotFoundError, match="File not found: .*"):
            router.run(sources=["non_existent_file.txt"])
