# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from haystack.components.converters.legacy_office import CONVERSION_MAPPING, LegacyOfficeConverter

INVERSE_CONVERSION_MAPPING = {v: k for k, v in CONVERSION_MAPPING.items()}


@pytest.fixture
def converter() -> LegacyOfficeConverter:
    return LegacyOfficeConverter()


@pytest.fixture
def mock_converter() -> Generator[LegacyOfficeConverter, None]:
    with patch("shutil.which", return_value="/usr/bin/soffice"):
        yield LegacyOfficeConverter()


class TestLegacyOfficeConverter:
    def test_init(self, mock_converter: LegacyOfficeConverter) -> None:
        assert isinstance(mock_converter, LegacyOfficeConverter)
        assert isinstance(mock_converter.soffice_path, str)

    def test_init_raises_when_soffice_not_found(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="LibreOffice"):
                LegacyOfficeConverter()

    def test_to_dict(self, mock_converter: LegacyOfficeConverter) -> None:
        data = mock_converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.legacy_office.LegacyOfficeConverter",
            "init_parameters": {},
        }

    def test_from_dict(self) -> None:
        data = {"type": "haystack.components.converters.legacy_office.LegacyOfficeConverter", "init_parameters": {}}
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LegacyOfficeConverter.from_dict(data)
        assert isinstance(converter.soffice_path, str)

    def test_run_unsupported_files(self, mock_converter: LegacyOfficeConverter) -> None:
        paths = ["test_file.pdf"]

        with pytest.raises(ValueError):
            mock_converter.run(paths, "test_path")

    def test_run_no_file(self, mock_converter: LegacyOfficeConverter) -> None:
        paths = ["test_file.doc"]

        with pytest.raises(FileNotFoundError):
            mock_converter.run(paths, "test_path")

    def test_run_bad_dir(self, mock_converter: LegacyOfficeConverter, test_files_path: Path) -> None:
        paths = [test_files_path / "docx" / "sample_doc.doc"]

        with pytest.raises(OSError):
            mock_converter.run(paths, "test_path")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run(self, converter: LegacyOfficeConverter, test_files_path: Path) -> None:
        paths = [
            test_files_path / "docx" / "sample_doc.doc",
            test_files_path / "pptx" / "sample_ppt.ppt",
            test_files_path / "xlsx" / "basic_tables_two_sheets.xls",
        ]

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            results = converter.run(paths, tmpdir_path)

            output_paths = results["output"]
            assert len(output_paths) == 3

            for input_file, output_file in zip(paths, output_paths, strict=True):
                assert output_file.is_file()
                assert INVERSE_CONVERSION_MAPPING[output_file.suffix[1:]] == input_file.suffix

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_run_async(self, converter: LegacyOfficeConverter, test_files_path: Path) -> None:
        paths = [
            test_files_path / "docx" / "sample_doc.doc",
            test_files_path / "pptx" / "sample_ppt.ppt",
            test_files_path / "xlsx" / "basic_tables_two_sheets.xls",
        ]

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            results = await converter.run_async(paths, tmpdir_path)

            output_paths = results["output"]
            assert len(output_paths) == 3

            for input_file, output_file in zip(paths, output_paths, strict=True):
                assert output_file.is_file()
                assert INVERSE_CONVERSION_MAPPING[output_file.suffix[1:]] == input_file.suffix
