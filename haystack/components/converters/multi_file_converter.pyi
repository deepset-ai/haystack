# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

from haystack import super_component
from haystack.core.super_component.super_component import _SuperComponent
from haystack.dataclasses import Document

@super_component
class MultiFileConverter(_SuperComponent):
    """
    A SuperComponent that converts multiple files into documents.

    This component can handle various file formats and convert them into Haystack Document objects.
    It supports multiple input files and returns a list of processed documents.
    """
    def __init__(self, *, encoding: str = "utf-8", json_content_key: str = "content") -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiFileConverter": ...
    def run(self, sources: List[str]) -> Dict[str, List[Document]]: ...
