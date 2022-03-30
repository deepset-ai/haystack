import pytest
from pathlib import Path
from haystack.nodes.file_classifier.file_type import FileTypeClassifier, DEFAULT_TYPES, FOLDER_NAMES
from .conftest import SAMPLES_PATH



def test_filetype_classifier_single_file(tmp_path):
    node = FileTypeClassifier()
    test_files = [tmp_path / f"test.{extension}" for extension in DEFAULT_TYPES]

    for edge_index, test_file in enumerate(test_files):
        output, edge = node.run(test_file)
        assert edge == f"output_{edge_index+1}"
        assert output == {"file_paths": [test_file]}


def test_filetype_classifier_many_files(tmp_path):
    node = FileTypeClassifier()

    for edge_index, extension in enumerate(DEFAULT_TYPES):
        test_files = [tmp_path / f"test_{idx}.{extension}" for idx in range(10)]

        output, edge = node.run(test_files)
        assert edge == f"output_{edge_index+1}"
        assert output == {"file_paths": test_files}


def test_filetype_classifier_many_files_mixed_extensions(tmp_path):
    node = FileTypeClassifier()
    test_files = [tmp_path / f"test.{extension}" for extension in DEFAULT_TYPES]

    with pytest.raises(ValueError):
        node.run(test_files)


def test_filetype_classifier_unsupported_extension(tmp_path):
    node = FileTypeClassifier()
    test_file = tmp_path / f"test.really_weird_extension"
    with pytest.raises(ValueError):
        node.run(test_file)


def test_filetype_classifier_custom_extensions(tmp_path):
    node = FileTypeClassifier(supported_types=["my_extension"])
    test_file = tmp_path / f"test.my_extension"
    output, edge = node.run(test_file)
    assert edge == f"output_1"
    assert output == {"file_paths": [test_file]}


def test_filetype_classifier_too_many_custom_extensions():
    with pytest.raises(ValueError):
        FileTypeClassifier(supported_types=[f"my_extension_{idx}" for idx in range(20)])


def test_filetype_classifier_duplicate_custom_extensions():
    with pytest.raises(ValueError):
        FileTypeClassifier(supported_types=[f"my_extension", "my_extension"])


def test_filetype_classifier_whithout_extension():
    node = FileTypeClassifier()
    test_files = []
    for folder_name in FOLDER_NAMES:
        test_files.append(SAMPLES_PATH / folder_name / "file_without_extension")

    for edge_index, test_file in enumerate(test_files):
        output, edge = node.run(test_file)
        print(edge, edge_index+1)
        assert edge == f"output_{edge_index+1}"
        assert output == {"file_paths": [test_file]}

