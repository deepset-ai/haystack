import sys

import pytest

from haystack.components.routers.file_type_router import FileTypeRouter
from haystack.dataclasses import ByteStream


@pytest.mark.skipif(
    sys.platform in ["win32", "cygwin"],
    reason="Can't run on Windows Github CI, need access to registry to get mime types",
)
class TestFileTypeRouter:
    def test_run(self, test_files_path):
        """
        Test if the component runs correctly in the simplest happy path.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "txt" / "doc_2.txt",
            test_files_path / "audio" / "the context for this answer is here.wav",
            test_files_path / "images" / "apple.jpg",
        ]

        router = FileTypeRouter(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        output = router.run(sources=file_paths)
        assert output
        assert len(output["text/plain"]) == 2
        assert len(output["audio/x-wav"]) == 1
        assert len(output["image/jpeg"]) == 1
        assert not output["unclassified"]

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
        mime_types = ["text/plain", "text/plain", "audio/x-wav", "image/jpeg"]
        # Convert file paths to ByteStream objects and set metadata
        byte_streams = []
        for path, mime_type in zip(file_paths, mime_types):
            stream = ByteStream(path.read_bytes())

            stream.meta["content_type"] = mime_type

            byte_streams.append(stream)

        # add unclassified ByteStream
        bs = ByteStream(b"unclassified content")
        bs.meta["content_type"] = "unknown_type"
        byte_streams.append(bs)

        router = FileTypeRouter(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        output = router.run(sources=byte_streams)
        assert output
        assert len(output["text/plain"]) == 2
        assert len(output["audio/x-wav"]) == 1
        assert len(output["image/jpeg"]) == 1
        assert len(output.get("unclassified")) == 1

    def test_run_with_bytestreams_and_file_paths(self, test_files_path):
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "audio" / "the context for this answer is here.wav",
            test_files_path / "txt" / "doc_2.txt",
            test_files_path / "images" / "apple.jpg",
            test_files_path / "markdown" / "sample.md",
        ]
        mime_types = ["text/plain", "audio/x-wav", "text/plain", "image/jpeg", "text/markdown"]
        byte_stream_sources = []
        for path, mime_type in zip(file_paths, mime_types):
            stream = ByteStream(path.read_bytes())
            stream.meta["content_type"] = mime_type
            byte_stream_sources.append(stream)

        mixed_sources = file_paths[:2] + byte_stream_sources[2:]

        router = FileTypeRouter(mime_types=["text/plain", "audio/x-wav", "image/jpeg", "text/markdown"])
        output = router.run(sources=mixed_sources)
        assert len(output["text/plain"]) == 2
        assert len(output["audio/x-wav"]) == 1
        assert len(output["image/jpeg"]) == 1
        assert len(output["text/markdown"]) == 1

    def test_no_files(self):
        """
        Test that the component runs correctly when no files are provided.
        """
        router = FileTypeRouter(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
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
        router = FileTypeRouter(mime_types=["text/plain"])
        output = router.run(sources=file_paths)
        assert len(output["text/plain"]) == 1
        assert "mp3" not in output
        assert len(output["unclassified"]) == 2
        assert str(output["unclassified"][0]).endswith("ignored.mp3")
        assert str(output["unclassified"][1]).endswith("this is the content of the document.wav")

    def test_no_extension(self, test_files_path):
        """
        Test that the component ignores files with no extension.
        """
        file_paths = [
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "txt" / "doc_2",
            test_files_path / "txt" / "doc_2.txt",
        ]
        router = FileTypeRouter(mime_types=["text/plain"])
        output = router.run(sources=file_paths)
        assert len(output["text/plain"]) == 2
        assert len(output["unclassified"]) == 1

    def test_unknown_mime_type(self):
        """
        Test that the component handles files with unknown mime types.
        """
        with pytest.raises(ValueError, match="Unknown mime type:"):
            FileTypeRouter(mime_types=["type_invalid"])
