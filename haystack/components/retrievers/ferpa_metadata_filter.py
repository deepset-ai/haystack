# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
FERPAMetadataFilter — FERPA-compliant document filter for Haystack RAG pipelines.

Enforces identity-scoped access control on retriever results before they reach
the LLM context window. Complies with 34 CFR § 99.31(a)(1) (legitimate educational
interest) and § 99.32 (record of disclosures).

Two filtering layers are applied in sequence:

1. **Identity pre-filter** — removes documents whose ``student_id`` or
   ``institution_id`` metadata does not match the authorized scope.
2. **Category authorization** — removes documents whose ``category`` is not in
   the authorized set (e.g., only ACADEMIC_RECORD is permitted, not DISCIPLINARY).

Documents that carry **no identity metadata** are treated as shared knowledge-base
content (course catalogues, policy handbooks, etc.) and pass through both layers
unchanged. This avoids inadvertently blocking general-purpose reference content.

Usage::

    from haystack import Pipeline
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.retrievers.ferpa_metadata_filter import FERPAMetadataFilter
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    doc_store = InMemoryDocumentStore()
    # ... add documents with meta = {student_id, institution_id, category} ...

    ferpa_filter = FERPAMetadataFilter(
        student_id="stu_001",
        institution_id="inst_abc",
        authorized_categories=["academic_record", "financial_aid"],
        requesting_user_id="advisor_007",
    )

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(doc_store))
    pipeline.add_component("ferpa_filter", ferpa_filter)
    pipeline.connect("retriever.documents", "ferpa_filter.documents")

    result = pipeline.run({"retriever": {"query_embedding": query_emb}})
    # result["ferpa_filter"]["documents"] contains only stu_001's authorized records

Regulatory basis:
    34 CFR § 99.31(a)(1) — access to educational records (legitimate educational interest)
    34 CFR § 99.32       — record of disclosures

See https://www2.ed.gov/policy/gen/guid/fpco/ferpa/index.html for FERPA guidance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class FERPADisclosureRecord:
    """
    Structured record of a FERPA disclosure event (34 CFR § 99.32).

    Attributes:
        student_id: Identifier of the student whose records were accessed.
        institution_id: Identifier of the institution.
        requesting_user_id: User or system that requested access.
        disclosed_at: UTC timestamp of the disclosure.
        total_retrieved: Number of documents returned by the retriever.
        total_disclosed: Number of documents that passed FERPA filtering.
        categories_disclosed: Record categories that were included in the result.
        pipeline_context: Label identifying the pipeline or workflow context.
    """

    student_id: str
    institution_id: str
    requesting_user_id: str
    disclosed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_retrieved: int = 0
    total_disclosed: int = 0
    categories_disclosed: list[str] = field(default_factory=list)
    pipeline_context: str = "haystack_pipeline"

    def to_log_entry(self) -> str:
        """Return a structured log string suitable for compliance audit logs."""
        return (
            f"[FERPA_DISCLOSURE] student_id={self.student_id!r} "
            f"institution_id={self.institution_id!r} "
            f"requesting_user_id={self.requesting_user_id!r} "
            f"disclosed_at={self.disclosed_at.isoformat()} "
            f"total_retrieved={self.total_retrieved} "
            f"total_disclosed={self.total_disclosed} "
            f"categories_disclosed={self.categories_disclosed!r} "
            f"pipeline_context={self.pipeline_context!r}"
        )


@component
class FERPAMetadataFilter:
    """
    Haystack component that enforces FERPA identity-scope filtering on retrieved
    documents before they enter the LLM context window.

    Integrates with any Haystack pipeline by consuming a ``documents`` input
    (from any retriever) and emitting only the authorized subset via the
    ``documents`` output. A ``disclosure_record`` output carries the structured
    34 CFR § 99.32 audit entry for downstream compliance logging.

    **Two enforcement layers:**

    1. *Identity pre-filter*: ``student_id`` and ``institution_id`` in
       ``Document.meta`` must match the authorized scope. If **either** field is
       absent, the document is treated as shared knowledge-base content and
       passes through.

    2. *Category authorization*: When a document carries a ``category`` field in
       ``Document.meta`` and ``authorized_categories`` is non-empty, the category
       must be in the authorized set.

    **Serialization:** The component is fully serializable via ``to_dict()``/
    ``from_dict()`` and works in serialized Haystack pipelines (YAML/JSON).

    Args:
        student_id: The authorized student identifier. Only documents matching
            this ID (or documents without any ``student_id`` metadata) pass
            through.
        institution_id: The authorized institution identifier. Only documents
            matching this ID (or documents without any ``institution_id``
            metadata) pass through.
        authorized_categories: List of record category strings that are permitted
            for this request (e.g. ``["academic_record", "financial_aid"]``).
            When empty, all categories are allowed.
        requesting_user_id: Identifier of the user or system making the request.
            Recorded in the disclosure audit log. Defaults to ``"unknown"``.
        student_id_field: Key in ``Document.meta`` for the student identifier.
            Default: ``"student_id"``.
        institution_id_field: Key in ``Document.meta`` for the institution identifier.
            Default: ``"institution_id"``.
        category_field: Key in ``Document.meta`` for the record category.
            Default: ``"category"``.
        pipeline_context: Label for the audit record (pipeline or workflow name).
            Default: ``"haystack_pipeline"``.
        raise_on_violation: When ``True``, raise ``PermissionError`` if any
            unauthorized document is detected. When ``False`` (default), silently
            remove unauthorized documents and emit a ``WARNING`` log.
    """

    def __init__(
        self,
        student_id: str,
        institution_id: str,
        authorized_categories: list[str] | None = None,
        requesting_user_id: str = "unknown",
        student_id_field: str = "student_id",
        institution_id_field: str = "institution_id",
        category_field: str = "category",
        pipeline_context: str = "haystack_pipeline",
        raise_on_violation: bool = False,
    ) -> None:
        self.student_id = student_id
        self.institution_id = institution_id
        self.authorized_categories = list(authorized_categories) if authorized_categories else []
        self.requesting_user_id = requesting_user_id
        self.student_id_field = student_id_field
        self.institution_id_field = institution_id_field
        self.category_field = category_field
        self.pipeline_context = pipeline_context
        self.raise_on_violation = raise_on_violation

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializes the component to a dictionary for pipeline persistence."""
        return default_to_dict(
            self,
            student_id=self.student_id,
            institution_id=self.institution_id,
            authorized_categories=self.authorized_categories,
            requesting_user_id=self.requesting_user_id,
            student_id_field=self.student_id_field,
            institution_id_field=self.institution_id_field,
            category_field=self.category_field,
            pipeline_context=self.pipeline_context,
            raise_on_violation=self.raise_on_violation,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FERPAMetadataFilter":
        """Deserializes the component from a dictionary."""
        return default_from_dict(cls, data)

    # ------------------------------------------------------------------
    # Core run method
    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document], disclosure_record=FERPADisclosureRecord)
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Filter documents to the authorized identity scope.

        Args:
            documents: Documents returned by an upstream retriever.  Each document
                may carry FERPA-relevant fields in its ``meta`` dict.

        Returns:
            A dict with two keys:

            - ``documents``: List of ``Document`` objects that passed both
              identity pre-filter and category authorization.
            - ``disclosure_record``: A ``FERPADisclosureRecord`` documenting this
              access event per 34 CFR § 99.32. Always present, even when zero
              documents pass.

        Raises:
            PermissionError: Only when ``raise_on_violation=True`` and at least
                one unauthorized document was detected.
        """
        total_retrieved = len(documents)
        authorized: list[Document] = []

        for doc in documents:
            if self._is_authorized(doc):
                authorized.append(doc)

        removed = total_retrieved - len(authorized)

        if removed > 0:
            if self.raise_on_violation:
                raise PermissionError(
                    f"FERPA violation: {removed} unauthorized document(s) blocked for "
                    f"student={self.student_id!r}, institution={self.institution_id!r}. "
                    "Check authorized_categories or identity scope."
                )
            logger.warning(
                "[FERPA_FILTER] Blocked %d unauthorized document(s) "
                "student_id=%r institution_id=%r total_retrieved=%d total_authorized=%d",
                removed,
                self.student_id,
                self.institution_id,
                total_retrieved,
                len(authorized),
            )

        categories_disclosed = self._extract_categories(authorized)
        record = FERPADisclosureRecord(
            student_id=self.student_id,
            institution_id=self.institution_id,
            requesting_user_id=self.requesting_user_id,
            total_retrieved=total_retrieved,
            total_disclosed=len(authorized),
            categories_disclosed=categories_disclosed,
            pipeline_context=self.pipeline_context,
        )
        logger.info(record.to_log_entry())

        return {"documents": authorized, "disclosure_record": record}

    # ------------------------------------------------------------------
    # Async variant
    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document], disclosure_record=FERPADisclosureRecord)
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Async variant of :meth:`run`. The filtering logic is CPU-bound and runs
        synchronously; this method exists to satisfy Haystack's async pipeline
        contract.

        Args:
            documents: Same as :meth:`run`.

        Returns:
            Same as :meth:`run`.
        """
        return self.run(documents)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_authorized(self, doc: Document) -> bool:
        """
        Return True if this document is within the authorized scope.

        A document passes when:
        - It carries no ``student_id`` metadata (shared knowledge-base content).
        - OR both ``student_id`` and ``institution_id`` match AND category passes.
        """
        meta = doc.meta or {}
        doc_student_id = meta.get(self.student_id_field, _SENTINEL)
        doc_institution_id = meta.get(self.institution_id_field, _SENTINEL)

        # Documents without identity metadata are shared content — pass through.
        if doc_student_id is _SENTINEL and doc_institution_id is _SENTINEL:
            return True

        # If only one identity field is present, treat as identity-tagged.
        if doc_student_id != self.student_id:
            return False
        if doc_institution_id is not _SENTINEL and doc_institution_id != self.institution_id:
            return False

        # Category check (only when authorized_categories is non-empty).
        if self.authorized_categories:
            doc_category = meta.get(self.category_field, _SENTINEL)
            if doc_category is not _SENTINEL and doc_category not in self.authorized_categories:
                return False

        return True

    def _extract_categories(self, documents: list[Document]) -> list[str]:
        """Return sorted unique category values from authorized documents."""
        categories: set[str] = set()
        for doc in documents:
            meta = doc.meta or {}
            cat = meta.get(self.category_field)
            if cat is not None:
                categories.add(str(cat))
        return sorted(categories)
