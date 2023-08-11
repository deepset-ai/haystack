import pytest

from haystack.preview.components.classifiers.file_classifier import FileTypeClassifier
from test.preview.components.base import BaseTestComponent
from test.conftest import preview_samples_path


class TestFileClassifier(BaseTestComponent):
    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        """
        Test if the component runs correctly in the simplest happy path.
        """
        file_paths = [
            preview_samples_path / "txt" / "doc_1.txt",
            preview_samples_path / "txt" / "doc_2.txt",
            preview_samples_path / "audio" / "the context for this answer is here.wav",
            preview_samples_path / "images" / "apple.jpg",
        ]

        classifier = FileTypeClassifier(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        output = classifier.run(paths=file_paths)
        assert output
        assert len(output["text_plain"]) == 2
        assert len(output["audio_x_wav"]) == 1
        assert len(output["image_jpeg"]) == 1
        assert not output["unclassified"]

    @pytest.mark.unit
    def test_no_files(self):
        """
        Test that the component runs correctly when no files are provided.
        """
        classifier = FileTypeClassifier(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        output = classifier.run(paths=[])
        assert not output

    @pytest.mark.unit
    def test_unlisted_extensions(self, preview_samples_path):
        """
        Test that the component correctly handles files with non specified mime types.
        """
        file_paths = [preview_samples_path / "txt" / "doc_1.txt", preview_samples_path / "audio" / "ignored.mp3"]
        classifier = FileTypeClassifier(mime_types=["text/plain"])
        output = classifier.run(paths=file_paths)
        assert len(output["text_plain"]) == 1
        assert "mp3" not in output
        assert len(output["unclassified"]) == 1
        assert str(output["unclassified"][0]).endswith("ignored.mp3")

    @pytest.mark.unit
    def test_no_extension(self, preview_samples_path):
        """
        Test that the component ignores files with no extension.
        """
        file_paths = [preview_samples_path / "txt" / "doc_1.txt", preview_samples_path / "txt" / "doc_2"]
        classifier = FileTypeClassifier(mime_types=["text/plain"])
        output = classifier.run(paths=file_paths)
        assert len(output["text_plain"]) == 1
        assert len(output["unclassified"]) == 1

    @pytest.mark.unit
    def test_unknown_mime_type(self):
        """
        Test that the component handles files with unknown mime types.
        """
        with pytest.raises(ValueError, match="The list of mime types"):
            FileTypeClassifier(mime_types=["type/invalid"])
