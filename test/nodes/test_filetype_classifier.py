import logging
import platform

import pytest

import haystack
from haystack.nodes.file_classifier.file_type import FileTypeClassifier, DEFAULT_TYPES

from ..conftest import SAMPLES_PATH


@pytest.mark.unit
def test_filetype_classifier_single_file(tmp_path):
    node = FileTypeClassifier()
    test_files = [tmp_path / f"test.{extension}" for extension in DEFAULT_TYPES]

    for edge_index, test_file in enumerate(test_files):
        output, edge = node.run(test_file)
        assert edge == f"output_{edge_index+1}"
        assert output == {"file_paths": [test_file]}


@pytest.mark.unit
def test_filetype_classifier_many_files(tmp_path):
    node = FileTypeClassifier()

    for edge_index, extension in enumerate(DEFAULT_TYPES):
        test_files = [tmp_path / f"test_{idx}.{extension}" for idx in range(10)]

        output, edge = node.run(test_files)
        assert edge == f"output_{edge_index+1}"
        assert output == {"file_paths": test_files}


@pytest.mark.unit
def test_filetype_classifier_many_files_mixed_extensions(tmp_path):
    node = FileTypeClassifier()
    test_files = [tmp_path / f"test.{extension}" for extension in DEFAULT_TYPES]

    with pytest.raises(ValueError):
        node.run(test_files)


@pytest.mark.unit
def test_filetype_classifier_unsupported_extension(tmp_path):
    node = FileTypeClassifier()
    test_file = tmp_path / f"test.really_weird_extension"
    with pytest.raises(ValueError):
        node.run(test_file)


@pytest.mark.unit
def test_filetype_classifier_custom_extensions(tmp_path):
    node = FileTypeClassifier(supported_types=["my_extension"])
    test_file = tmp_path / f"test.my_extension"
    output, edge = node.run(test_file)
    assert edge == f"output_1"
    assert output == {"file_paths": [test_file]}


@pytest.mark.unit
def test_filetype_classifier_duplicate_custom_extensions():
    with pytest.raises(ValueError):
        FileTypeClassifier(supported_types=[f"my_extension", "my_extension"])


@pytest.mark.unit
@pytest.mark.skipif(platform.system() in ["Windows", "Darwin"], reason="python-magic not available")
def test_filetype_classifier_text_files_without_extension():
    tested_types = ["docx", "html", "odt", "pdf", "pptx", "txt"]
    node = FileTypeClassifier(supported_types=tested_types)
    test_files = [SAMPLES_PATH / "extensionless_files" / f"{type_name}_file" for type_name in tested_types]

    for edge_index, test_file in enumerate(test_files):
        output, edge = node.run(test_file)
        assert edge == f"output_{edge_index+1}"
        assert output == {"file_paths": [test_file]}


@pytest.mark.unit
@pytest.mark.skipif(platform.system() in ["Windows", "Darwin"], reason="python-magic not available")
def test_filetype_classifier_other_files_without_extension():
    tested_types = ["gif", "jpg", "mp3", "png", "wav", "zip"]
    node = FileTypeClassifier(supported_types=tested_types)
    test_files = [SAMPLES_PATH / "extensionless_files" / f"{type_name}_file" for type_name in tested_types]

    for edge_index, test_file in enumerate(test_files):
        output, edge = node.run(test_file)
        assert edge == f"output_{edge_index+1}"
        assert output == {"file_paths": [test_file]}


@pytest.mark.unit
def test_filetype_classifier_text_files_without_extension_no_magic(monkeypatch, caplog):
    try:
        monkeypatch.delattr(haystack.nodes.file_classifier.file_type, "magic")
    except AttributeError:
        # magic not installed, even better
        pass

    node = FileTypeClassifier(supported_types=[""])

    with caplog.at_level(logging.ERROR):
        node.run(SAMPLES_PATH / "extensionless_files" / f"pdf_file")
        assert "'python-magic' is not installed" in caplog.text
