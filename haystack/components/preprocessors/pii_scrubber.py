# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from typing import Any, Literal

from haystack import Document, component, logging
from haystack.core.serialization import default_from_dict, default_to_dict

logger = logging.getLogger(__name__)

Entity = Literal["email", "phone", "credit_card"]

_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\+?1?[\s.\-]?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}",
    "credit_card": r"\b(?:\d[ \-]*?){13,16}\b",
}


def _luhn_valid(digits: str) -> bool:
    nums = [int(d) for d in digits]
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@component
class PIIScrubber:
    """
    Scrubs PII from Documents with stdlib regexes.

    Replaces emails, phone numbers, and credit card numbers with placeholders
    like `[EMAIL_1]` and records counts in `meta["pii_redactions"]`.
    Use it before sending retrieved documents to an LLM.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.preprocessors import PIIScrubber

    scrubber = PIIScrubber()
    result = scrubber.run(documents=[Document(content="Contact me at jane@example.com")])
    ```
    """

    def __init__(
        self,
        entities: list[Entity] | None = None,
        replacement_template: str = "[{entity}_{index}]",
        keep_id: bool = False,
    ) -> None:
        """
        Initializes the PIIScrubber component.

        :param entities: PII entities to scrub. Defaults to all supported.
        :param replacement_template: Template with `{entity}` and `{index}` placeholders.
        :param keep_id: If `True`, keeps the original document ID.
        """
        self.entities: list[Entity] = entities or ["email", "phone", "credit_card"]
        for entity in self.entities:
            if entity not in _PATTERNS:
                raise ValueError(f"Unsupported entity '{entity}'. Supported: {sorted(_PATTERNS)}.")
        if "{entity}" not in replacement_template or "{index}" not in replacement_template:
            raise ValueError("replacement_template must contain '{entity}' and '{index}'.")
        self.replacement_template = replacement_template
        self.keep_id = keep_id
        self._regexes = {entity: re.compile(_PATTERNS[entity]) for entity in self.entities}

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Scrubs PII from the documents.

        :param documents: List of Documents to scrub.
        :returns: A dictionary with the following key:
            - `documents`: List of scrubbed Documents.
        :raises TypeError: if documents is not a list of Documents.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("PIIScrubber expects a List of Documents as input.")

        counters: dict[str, int] = dict.fromkeys(self.entities, 0)
        scrubbed_docs = []
        for doc in documents:
            if doc.content is None:
                logger.warning("PIIScrubber only scrubs text documents but got None content.")
                scrubbed_docs.append(doc)
                continue
            text = doc.content
            doc_start = dict(counters)
            for entity in self.entities:

                def _replace(_match: re.Match, _entity: str = entity) -> str:
                    if _entity == "credit_card" and not _luhn_valid(re.sub(r"\D", "", _match.group(0))):
                        return _match.group(0)
                    counters[_entity] += 1
                    return self.replacement_template.format(entity=_entity.upper(), index=counters[_entity])

                text = self._regexes[entity].sub(_replace, text)
            redactions = []
            for entity in self.entities:
                made = counters[entity] - doc_start[entity]
                if made:
                    redactions.append(
                        {
                            "entity": entity,
                            "placeholder": self.replacement_template.format(
                                entity=entity.upper(), index=doc_start[entity] + 1
                            ),
                            "count": made,
                        }
                    )
            meta = deepcopy(doc.meta)
            if redactions:
                meta["pii_redactions"] = redactions
            scrubbed_docs.append(
                Document(
                    id=doc.id if self.keep_id else "",
                    content=text,
                    blob=doc.blob,
                    meta=meta,
                    score=doc.score,
                    embedding=doc.embedding,
                    sparse_embedding=doc.sparse_embedding,
                )
            )
        return {"documents": scrubbed_docs}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self, entities=self.entities, replacement_template=self.replacement_template, keep_id=self.keep_id
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PIIScrubber":
        """
        Deserializes the component from a dictionary.
        """
        return default_from_dict(cls, data)
