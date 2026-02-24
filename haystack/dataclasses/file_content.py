# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import mimetypes
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import filetype

from haystack import logging
from haystack.utils.dataclasses import _warn_on_inplace_mutation

logger = logging.getLogger(__name__)


@_warn_on_inplace_mutation
@dataclass
class FileContent:
    """
    The file content of a chat message.

    :param base64_data: A base64 string representing the file.
    :param mime_type: The MIME type of the file (e.g. "application/pdf").
        Providing this value is recommended, as most LLM providers require it.
        If not provided, the MIME type is guessed from the base64 string, which can be slow and not always reliable.
    :param filename: Optional filename of the file. Some LLM providers use this information.
    :param extra: Dictionary of extra information about the file. Can be used to store provider-specific information.
        To avoid serialization issues, values should be JSON serializable.
    :param validation: If True (default), a validation process is performed:
        - Check whether the base64 string is valid;
        - Guess the MIME type if not provided.
        Set to False to skip validation and speed up initialization.
    """

    base64_data: str
    mime_type: str | None = None
    filename: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    validation: bool = True

    def __post_init__(self):
        if not self.validation:
            return

        try:
            decoded_data = base64.b64decode(self.base64_data, validate=True)
        except Exception as e:
            raise ValueError("The base64 string is not valid") from e

        # mime_type is an important information, so we try to guess it if not provided
        if not self.mime_type:
            guess = filetype.guess(decoded_data)
            if guess:
                self.mime_type = guess.mime
            else:
                msg = (
                    "Failed to guess the MIME type of the file. Omitting the MIME type may result in "
                    "processing errors or incorrect handling of the file by LLM providers."
                )
                logger.warning(msg)

    def __repr__(self) -> str:
        """
        Return a string representation of the FileContent, truncating the base64_data to 100 bytes.
        """
        fields = []

        truncated_data = self.base64_data[:100] + "..." if len(self.base64_data) > 100 else self.base64_data
        fields.append(f"base64_data={truncated_data!r}")
        fields.append(f"mime_type={self.mime_type!r}")
        fields.append(f"filename={self.filename!r}")
        fields.append(f"extra={self.extra!r}")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}({fields_str})"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert FileContent into a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileContent":
        """
        Create an FileContent from a dictionary.
        """
        return FileContent(**data)

    @classmethod
    def from_file_path(
        cls, file_path: str | Path, *, filename: str | None = None, extra: dict[str, Any] | None = None
    ) -> "FileContent":
        """
        Create an FileContent object from a file path.

        :param file_path:
            The path to the file.
        :param filename:
            Optional file name. Some LLM providers use this information. If not provided, the filename is extracted
            from the file path.
        :param extra:
            Dictionary of extra information about the file. Can be used to store provider-specific information.
            To avoid serialization issues, values should be JSON serializable.

        :returns:
            An FileContent object.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        mime_type = mimetypes.guess_type(file_path.as_posix())[0]
        filename = filename or file_path.name

        with open(file_path, "rb") as f:
            data = f.read()

        return cls(
            base64_data=base64.b64encode(data).decode("utf-8"),
            mime_type=mime_type,
            filename=filename,
            extra=extra or {},
            validation=False,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        retry_attempts: int = 2,
        timeout: int = 10,
        filename: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "FileContent":
        """
        Create an FileContent object from a URL. The file is downloaded and converted to a base64 string.

        :param url:
            The URL of the file.
        :param retry_attempts:
            The number of times to retry to fetch the URL's content.
        :param timeout:
            Timeout in seconds for the request.
        :param filename:
            Optional filename of the file. Some LLM providers use this information. If not provided, the filename is
            extracted from the URL.
        :param extra:
            Dictionary of extra information about the file. Can be used to store provider-specific information.
            To avoid serialization issues, values should be JSON serializable.

        :returns:
            An FileContent object.
        """
        from haystack.components.fetchers.link_content import LinkContentFetcher

        fetcher = LinkContentFetcher(raise_on_failure=True, retry_attempts=retry_attempts, timeout=timeout)
        bytestream = fetcher.run(urls=[url])["streams"][0]

        mime_type = bytestream.mime_type
        data = bytestream.data

        if not filename:
            filename = os.path.basename(unquote(urlparse(url).path))

        return cls(
            base64_data=base64.b64encode(data).decode("utf-8"),
            mime_type=mime_type,
            filename=filename,
            extra=extra or {},
            validation=False,
        )
