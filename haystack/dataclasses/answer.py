# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pandas import DataFrame, read_json

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document


@runtime_checkable
@dataclass
class Answer(Protocol):
    data: Any
    query: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:  # noqa: D102
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Answer":  # noqa: D102
        ...


@dataclass
class ExtractedAnswer:
    query: str
    score: float
    data: Optional[str] = None
    document: Optional[Document] = None
    context: Optional[str] = None
    document_offset: Optional["Span"] = None
    context_offset: Optional["Span"] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Span:
        start: int
        end: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        document = self.document.to_dict(flatten=False) if self.document is not None else None
        document_offset = asdict(self.document_offset) if self.document_offset is not None else None
        context_offset = asdict(self.context_offset) if self.context_offset is not None else None
        return default_to_dict(
            self,
            data=self.data,
            query=self.query,
            document=document,
            context=self.context,
            score=self.score,
            document_offset=document_offset,
            context_offset=context_offset,
            meta=self.meta,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.
        :returns:
            Deserialized object.
        """
        init_params = data.get("init_parameters", {})
        if (doc := init_params.get("document")) is not None:
            data["init_parameters"]["document"] = Document.from_dict(doc)

        if (offset := init_params.get("document_offset")) is not None:
            data["init_parameters"]["document_offset"] = ExtractedAnswer.Span(**offset)

        if (offset := init_params.get("context_offset")) is not None:
            data["init_parameters"]["context_offset"] = ExtractedAnswer.Span(**offset)
        return default_from_dict(cls, data)


@dataclass
class ExtractedTableAnswer:
    query: str
    score: float
    data: Optional[str] = None
    document: Optional[Document] = None
    context: Optional[DataFrame] = None
    document_cells: List["Cell"] = field(default_factory=list)
    context_cells: List["Cell"] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Cell:
        row: int
        column: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        document = self.document.to_dict(flatten=False) if self.document is not None else None
        context = self.context.to_json() if self.context is not None else None
        document_cells = [asdict(c) for c in self.document_cells]
        context_cells = [asdict(c) for c in self.context_cells]
        return default_to_dict(
            self,
            data=self.data,
            query=self.query,
            document=document,
            context=context,
            score=self.score,
            document_cells=document_cells,
            context_cells=context_cells,
            meta=self.meta,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedTableAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.

        :returns:
            Deserialized object.
        """
        init_params = data.get("init_parameters", {})
        if (doc := init_params.get("document")) is not None:
            data["init_parameters"]["document"] = Document.from_dict(doc)

        if (context := init_params.get("context")) is not None:
            data["init_parameters"]["context"] = read_json(io.StringIO(context))

        if (cells := init_params.get("document_cells")) is not None:
            data["init_parameters"]["document_cells"] = [ExtractedTableAnswer.Cell(**c) for c in cells]

        if (cells := init_params.get("context_cells")) is not None:
            data["init_parameters"]["context_cells"] = [ExtractedTableAnswer.Cell(**c) for c in cells]
        return default_from_dict(cls, data)


@dataclass
class GeneratedAnswer:
    data: str
    query: str
    documents: List[Document]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        documents = [doc.to_dict(flatten=False) for doc in self.documents]
        return default_to_dict(self, data=self.data, query=self.query, documents=documents, meta=self.meta)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.

        :returns:
            Deserialized object.
        """
        init_params = data.get("init_parameters", {})
        if (documents := init_params.get("documents")) is not None:
            data["init_parameters"]["documents"] = [Document.from_dict(d) for d in documents]

        return default_from_dict(cls, data)
