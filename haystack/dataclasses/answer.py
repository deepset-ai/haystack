# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from haystack.dataclasses import ChatMessage, Document
from haystack.utils.dataclasses import _warn_on_inplace_mutation


@runtime_checkable
@dataclass
class Answer(Protocol):
    data: Any
    query: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Answer":  # noqa: D102
        ...


@_warn_on_inplace_mutation
@dataclass
class ExtractedAnswer:
    """
    Holds an answer extracted by an extractive Reader (query, score, text, and optional document/context).
    """

    query: str
    score: float
    data: str | None = None
    document: Document | None = None
    context: str | None = None
    document_offset: Optional["Span"] = None
    context_offset: Optional["Span"] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @_warn_on_inplace_mutation
    @dataclass
    class Span:
        start: int
        end: int

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        return {
            "data": self.data,
            "query": self.query,
            "document": self.document.to_dict(flatten=False) if self.document is not None else None,
            "context": self.context,
            "score": self.score,
            "document_offset": asdict(self.document_offset) if self.document_offset is not None else None,
            "context_offset": asdict(self.context_offset) if self.context_offset is not None else None,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.
        :returns:
            Deserialized object.
        """
        # backward compat: old format wrapped fields in init_parameters
        if "init_parameters" in data:
            data = data["init_parameters"]

        document = data.get("document")
        if document is not None:
            document = Document.from_dict(document)

        document_offset = data.get("document_offset")
        if document_offset is not None:
            document_offset = ExtractedAnswer.Span(**document_offset)

        context_offset = data.get("context_offset")
        if context_offset is not None:
            context_offset = ExtractedAnswer.Span(**context_offset)

        return cls(
            data=data.get("data"),
            query=data["query"],
            score=data["score"],
            document=document,
            context=data.get("context"),
            document_offset=document_offset,
            context_offset=context_offset,
            meta=data.get("meta", {}),
        )


@_warn_on_inplace_mutation
@dataclass
class GeneratedAnswer:
    """
    Holds a generated answer from a Generator (answer text, query, referenced documents, and metadata).
    """

    data: str
    query: str
    documents: list[Document]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the object to a dictionary.

        :returns:
            Serialized dictionary representation of the object.
        """
        # all_messages is either a list of ChatMessage objects or a list of strings
        meta = self.meta
        all_messages = meta.get("all_messages")
        if all_messages and isinstance(all_messages[0], ChatMessage):
            meta = {**meta, "all_messages": [msg.to_dict() for msg in all_messages]}

        return {
            "data": self.data,
            "query": self.query,
            "documents": [doc.to_dict(flatten=False) for doc in self.documents],
            "meta": meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedAnswer":
        """
        Deserialize the object from a dictionary.

        :param data:
            Dictionary representation of the object.

        :returns:
            Deserialized object.
        """
        # backward compatibility: old format wrapped fields in init_parameters
        if "init_parameters" in data:
            data = data["init_parameters"]

        documents = [Document.from_dict(d) for d in data.get("documents", [])]

        # copy to avoid mutating the caller's input dict when converting all_messages
        meta = dict(data.get("meta", {}))
        if (all_messages := meta.get("all_messages")) and isinstance(all_messages[0], dict):
            meta["all_messages"] = [ChatMessage.from_dict(m) for m in all_messages]

        return cls(data=data["data"], query=data["query"], documents=documents, meta=meta)
