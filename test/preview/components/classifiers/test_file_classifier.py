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

        classifier = FileTypeClassifier(extensions=["txt", "wav", "jpg"])
        data_input = classifier.input(file_paths)
        output = classifier.run(data=data_input)
        assert output
        assert len(output.txt) == 2
        assert len(output.wav) == 1
        assert len(output.jpg) == 1

    @pytest.mark.unit
    def test_no_files(self):
        """
        Test that the component runs correctly when no files are provided.
        """
        classifier = FileTypeClassifier(extensions=["txt", "wav", "jpg"])
        data_input = classifier.input([])
        output = classifier.run(data=data_input)
        assert not output.txt
        assert not output.wav
        assert not output.jpg

    @pytest.mark.unit
    def test_unlisted_extensions(self, preview_samples_path):
        """
        Test that the component ignores files with extensions not in the list.
        """
        file_paths = [preview_samples_path / "txt" / "doc_1.txt", preview_samples_path / "audio" / "ignored.mp3"]
        classifier = FileTypeClassifier(extensions=["txt"])
        data_input = classifier.input(file_paths)
        output = classifier.run(data=data_input)
        assert len(output.txt) == 1
        assert not hasattr(output, "mp3")

    @pytest.mark.unit
    def test_no_extension(self, preview_samples_path):
        """
        Test that the component handles files with no extensions.
        """
        file_paths = [preview_samples_path / "txt" / "doc_1.txt", preview_samples_path / "txt" / "doc_2"]
        classifier = FileTypeClassifier(extensions=["txt"])
        data_input = classifier.input(file_paths)
        output = classifier.run(data=data_input)
        assert len(output.txt) == 1
