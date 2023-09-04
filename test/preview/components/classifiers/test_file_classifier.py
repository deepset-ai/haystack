import sys

import pytest

from haystack.preview.components.classifiers.file_classifier import FileExtensionClassifier


@pytest.mark.skipif(
    sys.platform in ["win32", "cygwin"],
    reason="Can't run on Windows Github CI, need access to registry to get mime types",
)
class TestFileExtensionClassifier:
    @pytest.mark.unit
    def test_to_dict(self):
        component = FileExtensionClassifier(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        data = component.to_dict()
        assert data == {
            "type": "FileExtensionClassifier",
            "init_parameters": {"mime_types": ["text/plain", "audio/x-wav", "image/jpeg"]},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "FileExtensionClassifier",
            "init_parameters": {"mime_types": ["text/plain", "audio/x-wav", "image/jpeg"]},
        }
        component = FileExtensionClassifier.from_dict(data)
        assert component.mime_types == ["text/plain", "audio/x-wav", "image/jpeg"]

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

        classifier = FileExtensionClassifier(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        output = classifier.run(paths=file_paths)
        assert output
        assert len(output["text/plain"]) == 2
        assert len(output["audio/x-wav"]) == 1
        assert len(output["image/jpeg"]) == 1
        assert not output["unclassified"]

    @pytest.mark.unit
    def test_no_files(self):
        """
        Test that the component runs correctly when no files are provided.
        """
        classifier = FileExtensionClassifier(mime_types=["text/plain", "audio/x-wav", "image/jpeg"])
        output = classifier.run(paths=[])
        assert not output

    @pytest.mark.unit
    def test_unlisted_extensions(self, preview_samples_path):
        """
        Test that the component correctly handles files with non specified mime types.
        """
        file_paths = [
            preview_samples_path / "txt" / "doc_1.txt",
            preview_samples_path / "audio" / "ignored.mp3",
            preview_samples_path / "audio" / "this is the content of the document.wav",
        ]
        classifier = FileExtensionClassifier(mime_types=["text/plain"])
        output = classifier.run(paths=file_paths)
        assert len(output["text/plain"]) == 1
        assert "mp3" not in output
        assert len(output["unclassified"]) == 2
        assert str(output["unclassified"][0]).endswith("ignored.mp3")
        assert str(output["unclassified"][1]).endswith("this is the content of the document.wav")

    @pytest.mark.unit
    def test_no_extension(self, preview_samples_path):
        """
        Test that the component ignores files with no extension.
        """
        file_paths = [
            preview_samples_path / "txt" / "doc_1.txt",
            preview_samples_path / "txt" / "doc_2",
            preview_samples_path / "txt" / "doc_2.txt",
        ]
        classifier = FileExtensionClassifier(mime_types=["text/plain"])
        output = classifier.run(paths=file_paths)
        assert len(output["text/plain"]) == 2
        assert len(output["unclassified"]) == 1

    @pytest.mark.unit
    def test_unknown_mime_type(self):
        """
        Test that the component handles files with unknown mime types.
        """
        with pytest.raises(ValueError, match="Unknown mime type:"):
            FileExtensionClassifier(mime_types=["type_invalid"])
