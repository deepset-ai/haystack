# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DocumentStore


@component
class FERPAMetadataRetriever:
    """
    Retrieves documents that match both a ``student_id`` and an ``institution_id``
    metadata constraint, enforcing FERPA (Family Educational Rights and Privacy Act)
    identity boundaries in retrieval-augmented generation pipelines.

    This component is designed for higher-education AI systems where student records
    stored in the document store must not cross student or institution boundaries
    during retrieval. The identity filter is applied **at the document store query**,
    before any ranking or scoring occurs, so that unauthorized documents are never
    surfaced in the result set regardless of their semantic similarity to the query.

    The component builds a compound ``AND`` filter from the provided identity fields
    and delegates to the document store's ``filter_documents`` method, which applies
    the filter using the same metadata index used by all other retrievers. This makes
    the enforcement consistent with any other metadata-based access control already
    in place on the store.

    **FERPA compliance note:** This component enforces the identity isolation layer
    of a FERPA-compliant RAG pipeline. Callers are also responsible for:

    - Verifying that the ``student_id`` and ``institution_id`` values originate from
      an authenticated session, not from user-supplied query parameters.
    - Logging each retrieval event as required by 34 CFR § 99.32 (disclosure records).
    - Restricting retrieval to authorized document categories (e.g., excluding
      counseling notes if the session does not carry that permission).

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.retrievers import FERPAMetadataRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    docs = [
        Document(
            content="Alice is enrolled full-time, 15 credit hours.",
            meta={"student_id": "stu-alice", "institution_id": "univ-east", "category": "academic_record"},
        ),
        Document(
            content="Bob is enrolled part-time, 6 credit hours.",
            meta={"student_id": "stu-bob", "institution_id": "univ-east", "category": "academic_record"},
        ),
        Document(
            content="Carol is enrolled at West University.",
            meta={"student_id": "stu-carol", "institution_id": "univ-west", "category": "academic_record"},
        ),
    ]

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)

    retriever = FERPAMetadataRetriever(
        document_store=doc_store,
        student_id="stu-alice",
        institution_id="univ-east",
    )

    # Returns only Alice's document at univ-east.
    # Bob's document (different student) and Carol's document
    # (different institution) are excluded.
    result = retriever.run()
    print(result["documents"])
    ```

    The ``student_id`` and ``institution_id`` can also be supplied at run time,
    which allows the same component instance to serve different authenticated
    sessions in a pipeline:

    ```python
    retriever = FERPAMetadataRetriever(document_store=doc_store)
    result = retriever.run(student_id="stu-alice", institution_id="univ-east")
    ```
    """

    def __init__(
        self,
        document_store: DocumentStore,
        student_id: str | None = None,
        institution_id: str | None = None,
        student_id_field: str = "student_id",
        institution_id_field: str = "institution_id",
    ) -> None:
        """
        Create the FERPAMetadataRetriever component.

        :param document_store:
            An instance of a Document Store to use with the Retriever.
        :param student_id:
            The student identifier to filter on. Documents whose ``student_id``
            metadata field does not match this value are excluded. If ``None``,
            the value must be supplied at run time.
        :param institution_id:
            The institution identifier to filter on. Documents whose
            ``institution_id`` metadata field does not match this value are
            excluded. If ``None``, the value must be supplied at run time.
        :param student_id_field:
            The name of the metadata field that holds the student identifier.
            Defaults to ``"student_id"``. Override if your document store uses
            a different field name (e.g., ``"learner_id"`` or ``"user_id"``).
        :param institution_id_field:
            The name of the metadata field that holds the institution identifier.
            Defaults to ``"institution_id"``. Override if your document store uses
            a different field name (e.g., ``"org_id"`` or ``"tenant_id"``).
        """
        self.document_store = document_store
        self.student_id = student_id
        self.institution_id = institution_id
        self.student_id_field = student_id_field
        self.institution_id_field = institution_id_field

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"document_store": type(self.document_store).__name__}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store,
            student_id=self.student_id,
            institution_id=self.institution_id,
            student_id_field=self.student_id_field,
            institution_id_field=self.institution_id_field,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FERPAMetadataRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def _build_filter(self, student_id: str, institution_id: str) -> dict[str, Any]:
        """
        Build a compound AND filter matching both identity fields.

        :param student_id: The student identifier value to filter on.
        :param institution_id: The institution identifier value to filter on.
        :returns:
            A Haystack filter dict with an AND operator over both conditions.
        """
        return {
            "operator": "AND",
            "conditions": [
                {"field": f"meta.{self.student_id_field}", "operator": "==", "value": student_id},
                {"field": f"meta.{self.institution_id_field}", "operator": "==", "value": institution_id},
            ],
        }

    @component.output_types(documents=list[Document])
    def run(
        self,
        student_id: str | None = None,
        institution_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the FERPAMetadataRetriever.

        Retrieves documents from the document store that match both the
        ``student_id`` and ``institution_id`` identity constraints. Values
        supplied at run time take precedence over those provided at
        initialization.

        :param student_id:
            The student identifier to filter on at run time. If not supplied,
            the value from initialization is used. One of the two must be set.
        :param institution_id:
            The institution identifier to filter on at run time. If not
            supplied, the value from initialization is used. One of the two
            must be set.
        :raises ValueError:
            If neither an initialization-time nor a run-time value is available
            for ``student_id`` or ``institution_id``.
        :returns:
            A dict with a single key ``"documents"`` containing the list of
            documents that satisfy both identity constraints.
        """
        resolved_student_id = student_id or self.student_id
        resolved_institution_id = institution_id or self.institution_id

        if not resolved_student_id:
            msg = (
                "FERPAMetadataRetriever requires a student_id. "
                "Provide it at initialization or at run time."
            )
            raise ValueError(msg)

        if not resolved_institution_id:
            msg = (
                "FERPAMetadataRetriever requires an institution_id. "
                "Provide it at initialization or at run time."
            )
            raise ValueError(msg)

        filters = self._build_filter(
            student_id=resolved_student_id,
            institution_id=resolved_institution_id,
        )
        return {"documents": self.document_store.filter_documents(filters=filters)}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        student_id: str | None = None,
        institution_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously run the FERPAMetadataRetriever.

        See :meth:`run` for full parameter and return value documentation.

        :raises ValueError:
            If neither an initialization-time nor a run-time value is available
            for ``student_id`` or ``institution_id``.
        :returns:
            A dict with a single key ``"documents"`` containing the list of
            documents that satisfy both identity constraints.
        """
        resolved_student_id = student_id or self.student_id
        resolved_institution_id = institution_id or self.institution_id

        if not resolved_student_id:
            msg = (
                "FERPAMetadataRetriever requires a student_id. "
                "Provide it at initialization or at run time."
            )
            raise ValueError(msg)

        if not resolved_institution_id:
            msg = (
                "FERPAMetadataRetriever requires an institution_id. "
                "Provide it at initialization or at run time."
            )
            raise ValueError(msg)

        filters = self._build_filter(
            student_id=resolved_student_id,
            institution_id=resolved_institution_id,
        )
        # type: ignore[attr-defined] — filter_documents_async not in Protocol but exists in implementations
        out_documents = await self.document_store.filter_documents_async(filters=filters)  # type: ignore[attr-defined]
        return {"documents": out_documents}
