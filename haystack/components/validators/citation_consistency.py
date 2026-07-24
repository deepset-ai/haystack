# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)

CitationStatus = Literal["found", "misattributed", "not_found"]


@dataclass
class Citation:
    """
    A quote-style citation to be checked against retrieved documents.

    :param claim: The statement the citation is meant to support.
    :param document_id: The `id` of the `Document` the quote is attributed to.
    :param quote: The verbatim text the generator claims to have quoted from the document.
    :param status: Filled in by `CitationConsistencyChecker`: `"found"`, `"misattributed"`, or `"not_found"`.
    """

    claim: str
    document_id: str
    quote: str
    status: CitationStatus | None = field(default=None)


def _normalize(text: str) -> str:
    """
    Case/typography/whitespace-insensitive form for verbatim matching.

    Numbers and `%` are preserved because they are load-bearing in the kind of quantitative claims citations are
    usually asked to support.
    """
    text = text.lower()
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"[^a-z0-9%.]+", " ", text)
    return " ".join(text.split())


@component
class CitationConsistencyChecker:
    """
    Deterministically checks whether quote-style citations appear verbatim in the documents they cite.

    This is a runtime guardrail that sits after a Generator producing answers with explicit quotes. Unlike an
    LLM-based groundedness check, it uses **no model and no tokens**: it verifies, by normalized substring match,
    that each quoted passage actually exists in the retrieved `Document` it is attributed to. That catches the three
    failure modes an LLM judge is worst at, because they read as fluent and supportive:

    - **fabricated** — a plausible quote that appears in no retrieved document,
    - **frankenquote** — every word is real, but the sentence was stitched together and never written (the match is
      contiguous, not bag-of-words, so this fails), and
    - **misattributed** — a real quote lifted from a *different* retrieved document.

    Because it is pure string work, it does not degrade on long contexts ("lost in the middle") and needs no external
    API, which makes it a cheap first stage in front of an LLM groundedness checker rather than a replacement for one.

    Each citation is routed to the `consistent` output (its quote was found verbatim in the cited document) or to the
    `inconsistent` output (`misattributed` or `not_found`), with its `status` field filled in.

    ### Usage example

    ```python
    from haystack.components.validators import CitationConsistencyChecker, Citation
    from haystack.dataclasses import Document

    docs = [Document(id="d1", content="Body weight was unchanged in both arms of the trial.")]
    checker = CitationConsistencyChecker()

    result = checker.run(
        citations=[
            Citation(claim="Weight did not change.", document_id="d1", quote="Body weight was unchanged in both arms"),
            Citation(claim="Weight dropped sharply.", document_id="d1", quote="Body weight fell dramatically"),
        ],
        documents=docs,
    )
    # result["consistent"]   -> the first citation  (status="found")
    # result["inconsistent"] -> the second citation (status="not_found")
    ```

    The standalone, framework-agnostic version of this gate (plus an optional burden-of-proof LLM judge for the
    ambiguous "does this real quote actually *support* the claim?" case) lives at
    https://github.com/Palo-Alto-AI-Research-Lab/verbatim-citation-gate.
    """

    def __init__(self, treat_misattributed_as_consistent: bool = False) -> None:
        """
        Create a CitationConsistencyChecker.

        :param treat_misattributed_as_consistent: When `True`, a quote found verbatim in *some other* retrieved
            document (rather than the one cited) is routed to `consistent`. Defaults to `False`, so citing the wrong
            source is treated as an inconsistency.
        """
        self.treat_misattributed_as_consistent = treat_misattributed_as_consistent

    def to_dict(self) -> dict[str, Any]:
        """Serializes the component to a dictionary."""
        return default_to_dict(self, treat_misattributed_as_consistent=self.treat_misattributed_as_consistent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CitationConsistencyChecker":
        """Deserializes the component from a dictionary."""
        return default_from_dict(cls, data)

    def _gate(self, quote: str, document_id: str, docs: dict[str, str]) -> CitationStatus:
        q = _normalize(quote)
        if not q:
            return "not_found"
        cited = docs.get(document_id)
        if cited is not None and q in _normalize(cited):
            return "found"
        if any(q in _normalize(text) for doc_id, text in docs.items() if doc_id != document_id):
            return "misattributed"
        return "not_found"

    @component.output_types(consistent=list[Citation], inconsistent=list[Citation])
    def run(self, citations: list[Citation], documents: list[Document]) -> dict[str, list[Citation]]:
        """
        Check each citation against the retrieved documents.

        :param citations: The quote-style citations to verify.
        :param documents: The retrieved documents the citations should be grounded in. Documents with no `content`
            are ignored.
        :returns: A dictionary with:
            - `consistent`: citations whose quote was found verbatim in the cited document.
            - `inconsistent`: citations whose quote was `misattributed` or `not_found`.
            Each returned `Citation` has its `status` field set.
        """
        docs = {doc.id: doc.content for doc in documents if doc.content}
        consistent: list[Citation] = []
        inconsistent: list[Citation] = []

        for citation in citations:
            status = self._gate(citation.quote, citation.document_id, docs)
            checked = Citation(
                claim=citation.claim, document_id=citation.document_id, quote=citation.quote, status=status
            )
            is_consistent = status == "found" or (status == "misattributed" and self.treat_misattributed_as_consistent)
            (consistent if is_consistent else inconsistent).append(checked)

        if inconsistent:
            logger.debug(
                "CitationConsistencyChecker flagged {count} inconsistent citation(s).", count=len(inconsistent)
            )
        return {"consistent": consistent, "inconsistent": inconsistent}
