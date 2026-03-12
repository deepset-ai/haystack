# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from asyncio import create_subprocess_exec
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypedDict

from typing_extensions import Self

from haystack import component, default_from_dict, default_to_dict

CONVERSION_MAPPING = {".doc": "docx", ".ppt": "pptx", ".xls": "xlsx"}


class LegacyOfficeConverterOutput(TypedDict):
    output: list[Path]


@component
class LegacyOfficeConverter:
    def __init__(self) -> None:
        """
        Convert legacy office files (e.g. .doc, .ppt, .xls) to modern office files.

        Uses libreoffice's command line utility (soffice) to convert the files.

        ### Usage examples

        **Simple conversion:**
        ```python
        from pathlib import Path

        from haystack.components.converters import LegacyOfficeConverter

        # Setup paths
        output_directory = Path("modern_office_documents")
        output_directory.mkdir()

        # Convert documents
        converter = LegacyOfficeConverter()
        results = converter.run(sources=[Path("sample.doc")], output_directory=output_directory)
        print(results["output"])  # [Path('modern_office_documents/sample_docx.docx')]
        ```

        **Conversion pipeline:**
        ```python
        from pathlib import Path

        from haystack import Pipeline
        from haystack.components.converters import DOCXToDocument, LegacyOfficeConverter

        output_directory = Path("modern_office_documents")
        output_directory.mkdir(exist_ok=True)


        pipeline = Pipeline()
        pipeline.add_component("legacy_converter", LegacyOfficeConverter())
        pipeline.add_component("docx_converter", DOCXToDocument())

        pipeline.connect("legacy_converter.output", "docx_converter.sources")

        results = pipeline.run(
            {"legacy_converter": {"sources": [Path("sample_docx.doc")], "output_directory": output_directory}}
        )
        print(results["docx_converter"]["documents"])
        ```
        """
        soffice_path = shutil.which("soffice")
        if soffice_path is None:
            msg = """LibreOffice (soffice) is required but not installed or not in PATH.

- Install instructions: https://www.libreoffice.org/get-help/install-howto/"""
            raise FileNotFoundError(msg)

        self.soffice_path = soffice_path

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    @staticmethod
    def _get_conversion_args(source: str | Path, output_directory: str | Path) -> tuple[Path, list[str]]:
        """
        Validate source file and return the soffice arguments for conversion.

        :param source: Source file path.
        :param output_directory: Output directory to save converted files to.
        :returns: Tuple of (output_path, soffice_args).
        :raises ValueError: If the source file type is not supported.
        """
        source_path = Path(source)
        output_path = Path(output_directory)
        # Only .doc, .ppt and .xls files are supported
        if (source_suffix := source_path.suffix) not in CONVERSION_MAPPING:
            supported_types = ", ".join(CONVERSION_MAPPING)
            msg = f"{source_path} has extension {source_suffix}, but must be one of type {supported_types}"
            raise ValueError(msg)

        # Source file must exist
        if not source_path.is_file():
            msg = f"{source=} does not exist"
            raise FileNotFoundError(msg)

        # Output directory must exist and be writable
        if not output_path.is_dir() or not os.access(output_path, os.W_OK):
            msg = f"{output_directory=} must exist and be writable"
            raise OSError(msg)

        output_type = CONVERSION_MAPPING[source_suffix]
        args = ["soffice", "--headless", "--convert-to", output_type, "--outdir", str(output_directory), str(source)]
        return (output_path / source_path.name).with_suffix(f".{output_type}"), args

    @component.output_types(output=list[Path])
    def run(self, sources: Iterable[str | Path], output_directory: str | Path) -> LegacyOfficeConverterOutput:
        """
        Convert legacy office files (e.g. .doc, .ppt, .xls) to modern office files.

        :param sources:
            List of file paths.
        :param output_directory:
            Output directory to save converted files to.
        :returns:
            A dictionary with the following keys:
            - `output`: List of output file paths.
        """
        output_paths: list[Path] = []
        for source in sources:
            output_path, args = self._get_conversion_args(source, output_directory)

            subprocess.run(args, check=True)
            output_paths.append(output_path)

        return {"output": output_paths}

    @component.output_types(output=list[Path])
    async def run_async(
        self, sources: Iterable[str | Path], output_directory: str | Path
    ) -> LegacyOfficeConverterOutput:
        """
        Asynchronously convert legacy office files (e.g. .doc, .ppt, .xls) to modern office files.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param sources:
            List of file paths.
        :param output_directory:
            Output directory to save converted files to.
        :returns:
            A dictionary with the following keys:
            - `output`: List of output file paths.
        """
        output_paths: list[Path] = []
        for source in sources:
            output_path, args = self._get_conversion_args(source, output_directory)

            process = await create_subprocess_exec(*args)
            # Wait for process to complete as only one instance of soffice can occur at once
            await process.wait()

            output_paths.append(output_path)

        return {"output": output_paths}
