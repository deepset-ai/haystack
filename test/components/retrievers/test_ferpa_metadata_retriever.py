# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.components.retrievers.ferpa_metadata_retriever import FERPAMetadataRetriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.testing.factory import document_store_class


@pytest.fixture()
def sample_docs():
    """Documents representing records from two students at two institutions."""
    alice_east = [
        Document(
            content="Alice is enrolled full-time, 15 credit hours.",
            meta={"student_id": "stu-alice", "institution_id": "univ-east", "category": "academic_record"},
        ),
        Document(
            content="Alice: tuition balance $0.",
            meta={"student_id": "stu-alice", "institution_id": "univ-east", "category": "financial_record"},
        ),
    ]
    bob_east = [
        Document(
            content="Bob is enrolled part-time, 6 credit hours.",
            meta={"student_id": "stu-bob", "institution_id": "univ-east", "category": "academic_record"},
        ),
    ]
    carol_west = [
        Document(
            content="Carol is enrolled at West University, 12 credit hours.",
            meta={"student_id": "stu-carol", "institution_id": "univ-west", "category": "academic_record"},
        ),
    ]
    all_docs = alice_east + bob_east + carol_west
    return {
        "alice_east": alice_east,
        "bob_east": bob_east,
        "carol_west": carol_west,
        "all_docs": all_docs,
    }


@pytest.fixture()
def sample_document_store(sample_docs):
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(sample_docs["all_docs"])
    return doc_store


class TestFERPAMetadataRetriever:
    @classmethod
    def _documents_equal(cls, docs1: list[Document], docs2: list[Document]) -> bool:
        docs1.sort(key=lambda x: x.id)
        docs2.sort(key=lambda x: x.id)
        return docs1 == docs2

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_init_default(self, sample_document_store):
        retriever = FERPAMetadataRetriever(sample_document_store)
        assert retriever.student_id is None
        assert retriever.institution_id is None
        assert retriever.student_id_field == "student_id"
        assert retriever.institution_id_field == "institution_id"

    def test_init_with_identity(self, sample_document_store):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-alice",
            institution_id="univ-east",
        )
        assert retriever.student_id == "stu-alice"
        assert retriever.institution_id == "univ-east"

    def test_init_with_custom_field_names(self, sample_document_store):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-alice",
            institution_id="univ-east",
            student_id_field="learner_id",
            institution_id_field="org_id",
        )
        assert retriever.student_id_field == "learner_id"
        assert retriever.institution_id_field == "org_id"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_to_dict(self):
        FakeStore = document_store_class("FakeStore", bases=(InMemoryDocumentStore,))
        doc_store = FakeStore()
        doc_store.to_dict = lambda: {"type": "FakeStore", "init_parameters": {}}

        component = FERPAMetadataRetriever(
            document_store=doc_store,
            student_id="stu-alice",
            institution_id="univ-east",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.retrievers.ferpa_metadata_retriever.FERPAMetadataRetriever",
            "init_parameters": {
                "document_store": {"type": "FakeStore", "init_parameters": {}},
                "student_id": "stu-alice",
                "institution_id": "univ-east",
                "student_id_field": "student_id",
                "institution_id_field": "institution_id",
            },
        }

    def test_from_dict(self):
        valid_data = {
            "type": "haystack.components.retrievers.ferpa_metadata_retriever.FERPAMetadataRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                },
                "student_id": "stu-alice",
                "institution_id": "univ-east",
                "student_id_field": "student_id",
                "institution_id_field": "institution_id",
            },
        }
        component = FERPAMetadataRetriever.from_dict(valid_data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.student_id == "stu-alice"
        assert component.institution_id == "univ-east"

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def test_build_filter_structure(self, sample_document_store):
        retriever = FERPAMetadataRetriever(sample_document_store)
        f = retriever._build_filter("stu-alice", "univ-east")
        assert f["operator"] == "AND"
        conditions = {c["field"]: c for c in f["conditions"]}
        assert "meta.student_id" in conditions
        assert conditions["meta.student_id"]["value"] == "stu-alice"
        assert "meta.institution_id" in conditions
        assert conditions["meta.institution_id"]["value"] == "univ-east"

    def test_build_filter_custom_field_names(self, sample_document_store):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id_field="learner_id",
            institution_id_field="org_id",
        )
        f = retriever._build_filter("stu-alice", "univ-east")
        fields = [c["field"] for c in f["conditions"]]
        assert "meta.learner_id" in fields
        assert "meta.org_id" in fields

    # ------------------------------------------------------------------
    # Retrieval — identity enforcement
    # ------------------------------------------------------------------

    def test_retrieval_with_init_identity(self, sample_document_store, sample_docs):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-alice",
            institution_id="univ-east",
        )
        result = retriever.run()

        assert "documents" in result
        assert TestFERPAMetadataRetriever._documents_equal(
            result["documents"], sample_docs["alice_east"]
        )

    def test_retrieval_with_runtime_identity(self, sample_document_store, sample_docs):
        retriever = FERPAMetadataRetriever(sample_document_store)
        result = retriever.run(student_id="stu-alice", institution_id="univ-east")

        assert "documents" in result
        assert TestFERPAMetadataRetriever._documents_equal(
            result["documents"], sample_docs["alice_east"]
        )

    def test_runtime_identity_overrides_init(self, sample_document_store, sample_docs):
        # Init with Alice; run with Bob — Bob's docs should be returned.
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-alice",
            institution_id="univ-east",
        )
        result = retriever.run(student_id="stu-bob", institution_id="univ-east")

        assert "documents" in result
        assert TestFERPAMetadataRetriever._documents_equal(
            result["documents"], sample_docs["bob_east"]
        )

    def test_cross_student_isolation(self, sample_document_store, sample_docs):
        """Bob's records must not be returned when querying for Alice."""
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-alice",
            institution_id="univ-east",
        )
        result = retriever.run()

        returned_ids = {doc.meta["student_id"] for doc in result["documents"]}
        assert "stu-bob" not in returned_ids
        assert "stu-carol" not in returned_ids

    def test_cross_institution_isolation(self, sample_document_store, sample_docs):
        """
        Carol's records at univ-west must not appear when querying the univ-east
        store, even if there happens to be a student at univ-east with the same ID.
        """
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-carol",
            institution_id="univ-east",  # note: carol is at univ-west
        )
        result = retriever.run()

        assert result["documents"] == []

    def test_no_results_for_unknown_student(self, sample_document_store):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-unknown",
            institution_id="univ-east",
        )
        result = retriever.run()
        assert result["documents"] == []

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_missing_student_id_raises(self, sample_document_store):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            institution_id="univ-east",
        )
        with pytest.raises(ValueError, match="student_id"):
            retriever.run()

    def test_missing_institution_id_raises(self, sample_document_store):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-alice",
        )
        with pytest.raises(ValueError, match="institution_id"):
            retriever.run()

    def test_missing_both_raises(self, sample_document_store):
        retriever = FERPAMetadataRetriever(sample_document_store)
        with pytest.raises(ValueError):
            retriever.run()

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    @pytest.mark.integration
    def test_run_in_pipeline(self, sample_document_store, sample_docs):
        retriever = FERPAMetadataRetriever(
            sample_document_store,
            student_id="stu-alice",
            institution_id="univ-east",
        )

        pipeline = Pipeline()
        pipeline.add_component("ferpa_retriever", retriever)
        result = pipeline.run(data={"ferpa_retriever": {}})

        assert "ferpa_retriever" in result
        docs = result["ferpa_retriever"]["documents"]
        assert TestFERPAMetadataRetriever._documents_equal(docs, sample_docs["alice_east"])

    @pytest.mark.integration
    def test_run_in_pipeline_with_runtime_identity(self, sample_document_store, sample_docs):
        retriever = FERPAMetadataRetriever(sample_document_store)

        pipeline = Pipeline()
        pipeline.add_component("ferpa_retriever", retriever)

        result = pipeline.run(
            data={"ferpa_retriever": {"student_id": "stu-bob", "institution_id": "univ-east"}}
        )
        docs = result["ferpa_retriever"]["documents"]
        assert TestFERPAMetadataRetriever._documents_equal(docs, sample_docs["bob_east"])
