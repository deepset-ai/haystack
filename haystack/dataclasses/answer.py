from typing import Any, Dict, List, Optional
from dataclasses import asdict, dataclass, fields
from haystack.dataclasses.document import Document


@dataclass(frozen=True)
class Answer:
    data: Any
    query: str
    metadata: Dict[str, Any]

    def to_dict(self, flatten=True) -> Dict[str, Any]:
        """
        Converts Answer into a dictionary.

        :param flatten: Whether to flatten `metadata` field or not. Defaults to `True`.
        """
        data = asdict(self)
        if flatten:
            meta = data.pop("metadata")
            return {**data, **meta}
        return data

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "Answer":
        """
        Creates a new Answer object from a dictionary.
        """
        # Store metadata for a moment while we try un-flattening allegedly flatten metadata.
        # We don't expect both a `metadata=` keyword and flatten metadata keys so we'll raise a
        # ValueError later if this is the case.
        meta = data_dict.pop("metadata", {})

        # Unflatten metadata if it was flattened. We assume any keyword argument that's not
        # a document field is a metadata key.
        flatten_meta = {}
        answer_fields = [f.name for f in fields(cls)]
        for key in list(data_dict.keys()):
            if key not in answer_fields:
                flatten_meta[key] = data_dict.pop(key)

        # We don't support passing both flatten keys and the `metadata` keyword parameter
        if meta and flatten_meta:
            raise ValueError(
                "You can pass either the 'metadata' parameter or flattened metadata keys as keyword arguments, "
                "but currently you're passing both. Pass either the 'metadata' parameter or flattened metadata keys."
            )

        # Finally put back all the metadata
        return cls(**data_dict, metadata={**meta, **flatten_meta})


@dataclass(frozen=True)
class ExtractedAnswer(Answer):
    data: Optional[str]
    document: Optional[Document]
    probability: float
    start: Optional[int] = None
    end: Optional[int] = None

    def to_dict(self, flatten=True) -> Dict[str, Any]:
        """
        Converts ExtractedAnswer into a dictionary.

        :param flatten: Whether to flatten `metadata` field or not. Defaults to `True`.
        """
        data = super().to_dict(flatten)
        if self.document:
            data["document"] = self.document.to_dict(flatten)
        return data

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "ExtractedAnswer":
        """
        Creates a new ExtractedAnswer object from a dictionary.
        """
        # Convert document to dictionary, if present
        if data_dict["document"]:
            doc = data_dict.pop("document")
            data_dict["document"] = Document.from_dict(doc)

        # Store metadata for a moment while we try un-flattening allegedly flatten metadata.
        # We don't expect both a `metadata=` keyword and flatten metadata keys so we'll raise a
        # ValueError later if this is the case.
        meta = data_dict.pop("metadata", {})

        # Unflatten metadata if it was flattened. We assume any keyword argument that's not
        # a document field is a metadata key.
        flatten_meta = {}
        answer_fields = [f.name for f in fields(cls)]
        for key in list(data_dict.keys()):
            if key not in answer_fields:
                flatten_meta[key] = data_dict.pop(key)

        # We don't support passing both flatten keys and the `metadata` keyword parameter
        if meta and flatten_meta:
            raise ValueError(
                "You can pass either the 'metadata' parameter or flattened metadata keys as keyword arguments, "
                "but currently you're passing both. Pass either the 'metadata' parameter or flattened metadata keys."
            )

        # Finally put back all the metadata
        return cls(**data_dict, metadata={**meta, **flatten_meta})


@dataclass(frozen=True)
class GeneratedAnswer(Answer):
    data: str
    documents: List[Document]

    def to_dict(self, flatten=True) -> Dict[str, Any]:
        """
        Converts GeneratedAnswer into a dictionary.

        :param flatten: Whether to flatten `metadata` field or not. Defaults to `True`.
        """
        data = super().to_dict(flatten)
        data["documents"] = [doc.to_dict(flatten) for doc in self.documents]
        return data

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "GeneratedAnswer":
        """
        Creates a new GeneratedAnswer object from a dictionary.
        """
        # Convert Documents to dictionaries
        documents = data_dict.pop("documents", {})
        data_dict["documents"] = [Document.from_dict(doc) for doc in documents]

        # Store metadata for a moment while we try un-flattening allegedly flatten metadata.
        # We don't expect both a `metadata=` keyword and flatten metadata keys so we'll raise a
        # ValueError later if this is the case.
        meta = data_dict.pop("metadata", {})

        # Unflatten metadata if it was flattened. We assume any keyword argument that's not
        # a document field is a metadata key.
        flatten_meta = {}
        answer_fields = [f.name for f in fields(cls)]
        for key in list(data_dict.keys()):
            if key not in answer_fields:
                flatten_meta[key] = data_dict.pop(key)

        # We don't support passing both flatten keys and the `metadata` keyword parameter
        if meta and flatten_meta:
            raise ValueError(
                "You can pass either the 'metadata' parameter or flattened metadata keys as keyword arguments, "
                "but currently you're passing both. Pass either the 'metadata' parameter or flattened metadata keys."
            )

        # Finally put back all the metadata
        return cls(**data_dict, metadata={**meta, **flatten_meta})
