# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)

_SRT_TIME = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})\s*-->\s*"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2})[,.](?P<ms2>\d{3})"
)
_VTT_TIME = re.compile(
    r"(?:(?P<h>\d{2}):)?(?P<m>\d{2}):(?P<s>\d{2})[.](?P<ms>\d{3})\s*-->\s*"
    r"(?:(?P<h2>\d{2}):)?(?P<m2>\d{2}):(?P<s2>\d{2})[.](?P<ms2>\d{3})"
)
_INLINE_TAG = re.compile(r"</?(?:c|i|b|u|v|lang|ruby|rt)(?:\s[^>]*)?>", re.IGNORECASE)
_CUE_NUMBER = re.compile(r"^\d+$")


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    hours = int(h or 0)
    return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _strip_inline_tags(text: str) -> str:
    return _INLINE_TAG.sub("", text).strip()


def _parse_cues(text: str) -> list[tuple[float, float, str]]:
    """Parse SRT or WebVTT text into (start, end, cue_text) triples."""
    # Normalize newlines and drop a leading UTF-8 BOM / WEBVTT header noise.
    content = text.replace("\r\n", "\n").replace("\r", "\n")
    if content.lstrip().startswith("WEBVTT"):
        # Drop header until the first blank line.
        parts = content.split("\n\n", 1)
        content = parts[1] if len(parts) == 2 else ""

    cues: list[tuple[float, float, str]] = []
    blocks = re.split(r"\n{2,}", content.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # Skip NOTE / STYLE / REGION blocks in VTT
        first = lines[0]
        if first.upper().startswith(("NOTE", "STYLE", "REGION")):
            continue
        # Optional cue identifier / SRT index
        if _CUE_NUMBER.match(first) or ("-->" not in first and len(lines) > 1 and "-->" in lines[1]):
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        timing = lines[0]
        match = _SRT_TIME.search(timing) or _VTT_TIME.search(timing)
        if not match:
            continue
        start = _to_seconds(match.group("h"), match.group("m"), match.group("s"), match.group("ms"))
        end = _to_seconds(match.group("h2"), match.group("m2"), match.group("s2"), match.group("ms2"))
        body_lines = [_strip_inline_tags(ln) for ln in lines[1:]]
        body = "\n".join(ln for ln in body_lines if ln)
        if not body:
            continue
        cues.append((start, end, body))
    return cues


@component
class SubtitleToDocument:
    """
    Converts SubRip (`.srt`) and WebVTT (`.vtt`) subtitle files into Documents.

    Cue numbers, timing lines, `NOTE` blocks and common inline tags are stripped.
    Consecutive cues are grouped into Documents (default 10 cues per Document) with
    `start_time` / `end_time` metadata in seconds so retrieved chunks stay traceable
    to a position in the media.

    Do not register this converter in `MultiFileConverter` yet — routing `.srt` away
    from `TextFileToDocument` is a separate, backwards-compatible follow-up.

    ### Usage example

    ```python
    from haystack.components.converters.subtitle import SubtitleToDocument

    converter = SubtitleToDocument()
    results = converter.run(sources=["clip.srt"])
    documents = results["documents"]
    print(documents[0].meta["start_time"], documents[0].content)
    ```
    """

    def __init__(
        self,
        encoding: str = "utf-8-sig",
        store_full_path: bool = False,
        cues_per_document: int | None = 10,
    ) -> None:
        """
        Creates a SubtitleToDocument component.

        :param encoding:
            The encoding of the subtitle files to convert.
            If the encoding is specified in the metadata of a source ByteStream,
            it overrides this value.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        :param cues_per_document:
            How many consecutive cues to merge into one Document. Defaults to 10.
            Pass `None` to emit a single Document per file.
        """
        if cues_per_document is not None and cues_per_document < 1:
            raise ValueError("cues_per_document must be >= 1 or None")
        self.encoding = encoding
        self.store_full_path = store_full_path
        self.cues_per_document = cues_per_document

    @component.output_types(documents=list[Document])
    def run(
        self, sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, list[Document]]:
        """
        Converts subtitle files to documents.

        :param sources:
            List of `.srt` / `.vtt` file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources as they're zipped together.
            For ByteStream objects, their `meta` is added to the output documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of converted documents.
        """
        documents: list[Document] = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                encoding = bytestream.meta.get("encoding", self.encoding)
                text = bytestream.data.decode(encoding)
            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e
                )
                continue

            cues = _parse_cues(text)
            if not cues:
                logger.warning(
                    "No valid subtitle cues found in {source}. Skipping it.",
                    source=source,
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)

            chunk_size = self.cues_per_document if self.cues_per_document is not None else len(cues)
            for i in range(0, len(cues), chunk_size):
                group = cues[i : i + chunk_size]
                content = "\n".join(cue_text for _, _, cue_text in group)
                doc_meta = {
                    **merged_metadata,
                    "start_time": group[0][0],
                    "end_time": group[-1][1],
                }
                documents.append(Document(content=content, meta=doc_meta))

        return {"documents": documents}
