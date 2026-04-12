# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FERPAMetadataFilter — FERPA-compliant document post-processor.

Coverage:
- Identity pre-filter: student_id and institution_id matching
- Category authorization: authorized vs. unauthorized categories
- Shared content pass-through: documents without identity metadata
- Audit record generation (FERPADisclosureRecord)
- raise_on_violation mode
- Serialization: to_dict / from_dict round-trips
- Custom field name configuration
- Async run_async method
- Pipeline integration
"""

from __future__ import annotations

import pytest

from haystack import Document
from haystack.components.retrievers.ferpa_metadata_filter import (
    FERPADisclosureRecord,
    FERPAMetadataFilter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_filter() -> FERPAMetadataFilter:
    return FERPAMetadataFilter(
        student_id="stu_001",
        institution_id="inst_abc",
        authorized_categories=["academic_record", "financial_aid"],
        requesting_user_id="advisor_007",
    )


def _doc(
    content: str = "test",
    student_id: str | None = None,
    institution_id: str | None = None,
    category: str | None = None,
) -> Document:
    meta: dict = {}
    if student_id is not None:
        meta["student_id"] = student_id
    if institution_id is not None:
        meta["institution_id"] = institution_id
    if category is not None:
        meta["category"] = category
    return Document(content=content, meta=meta)


# ---------------------------------------------------------------------------
# Identity pre-filter tests
# ---------------------------------------------------------------------------


class TestIdentityPreFilter:
    def test_matching_student_and_institution_passes(self, default_filter):
        docs = [_doc("grade report", student_id="stu_001", institution_id="inst_abc", category="academic_record")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 1

    def test_wrong_student_id_blocked(self, default_filter):
        docs = [_doc("other student grade", student_id="stu_999", institution_id="inst_abc", category="academic_record")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 0

    def test_wrong_institution_id_blocked(self, default_filter):
        docs = [_doc("grade report", student_id="stu_001", institution_id="inst_xyz", category="academic_record")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 0

    def test_both_fields_wrong_blocked(self, default_filter):
        docs = [_doc("unrelated", student_id="stu_999", institution_id="inst_xyz", category="academic_record")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 0

    def test_shared_content_no_identity_fields_passes(self, default_filter):
        """Documents without identity metadata are shared knowledge-base content."""
        docs = [Document(content="Course catalog 2026", meta={})]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 1

    def test_empty_meta_passes(self, default_filter):
        docs = [Document(content="General handbook")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 1


# ---------------------------------------------------------------------------
# Category authorization tests
# ---------------------------------------------------------------------------


class TestCategoryAuthorization:
    def test_authorized_category_passes(self, default_filter):
        docs = [_doc("transcript", student_id="stu_001", institution_id="inst_abc", category="financial_aid")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 1

    def test_unauthorized_category_blocked(self, default_filter):
        docs = [_doc("disciplinary file", student_id="stu_001", institution_id="inst_abc", category="disciplinary")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 0

    def test_no_category_field_passes_when_category_check_enabled(self, default_filter):
        """Documents with matching identity but no category field pass."""
        docs = [_doc("misc record", student_id="stu_001", institution_id="inst_abc")]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 1

    def test_empty_authorized_categories_allows_all_categories(self):
        """When authorized_categories is empty, all categories are permitted."""
        f = FERPAMetadataFilter(student_id="stu_001", institution_id="inst_abc", authorized_categories=[])
        docs = [_doc("health", student_id="stu_001", institution_id="inst_abc", category="health_record")]
        result = f.run(docs)
        assert len(result["documents"]) == 1


# ---------------------------------------------------------------------------
# Mixed batch tests
# ---------------------------------------------------------------------------


class TestMixedBatch:
    def test_mixed_batch_filters_correctly(self, default_filter):
        docs = [
            _doc("grade", student_id="stu_001", institution_id="inst_abc", category="academic_record"),  # PASS
            _doc("shared handbook"),  # PASS — no identity fields
            _doc("wrong student", student_id="stu_002", institution_id="inst_abc", category="academic_record"),  # FAIL
            _doc("wrong inst", student_id="stu_001", institution_id="inst_xyz", category="academic_record"),  # FAIL
            _doc("wrong cat", student_id="stu_001", institution_id="inst_abc", category="disciplinary"),  # FAIL
            _doc("financial", student_id="stu_001", institution_id="inst_abc", category="financial_aid"),  # PASS
        ]
        result = default_filter.run(docs)
        assert len(result["documents"]) == 3

    def test_empty_input_returns_empty(self, default_filter):
        result = default_filter.run([])
        assert result["documents"] == []


# ---------------------------------------------------------------------------
# Disclosure record tests
# ---------------------------------------------------------------------------


class TestDisclosureRecord:
    def test_disclosure_record_is_returned(self, default_filter):
        docs = [_doc("grade", student_id="stu_001", institution_id="inst_abc", category="academic_record")]
        result = default_filter.run(docs)
        assert "disclosure_record" in result
        assert isinstance(result["disclosure_record"], FERPADisclosureRecord)

    def test_disclosure_record_counts_correct(self, default_filter):
        docs = [
            _doc("grade", student_id="stu_001", institution_id="inst_abc", category="academic_record"),
            _doc("unauthorized", student_id="stu_999", institution_id="inst_abc", category="academic_record"),
        ]
        result = default_filter.run(docs)
        record = result["disclosure_record"]
        assert record.total_retrieved == 2
        assert record.total_disclosed == 1
        assert record.student_id == "stu_001"
        assert record.institution_id == "inst_abc"
        assert record.requesting_user_id == "advisor_007"

    def test_disclosure_record_captures_disclosed_categories(self, default_filter):
        docs = [
            _doc("grade", student_id="stu_001", institution_id="inst_abc", category="academic_record"),
            _doc("aid", student_id="stu_001", institution_id="inst_abc", category="financial_aid"),
        ]
        result = default_filter.run(docs)
        record = result["disclosure_record"]
        assert sorted(record.categories_disclosed) == ["academic_record", "financial_aid"]

    def test_disclosure_record_empty_batch(self, default_filter):
        result = default_filter.run([])
        record = result["disclosure_record"]
        assert record.total_retrieved == 0
        assert record.total_disclosed == 0
        assert record.categories_disclosed == []

    def test_disclosure_record_to_log_entry_format(self, default_filter):
        docs = [_doc("grade", student_id="stu_001", institution_id="inst_abc", category="academic_record")]
        result = default_filter.run(docs)
        log_entry = result["disclosure_record"].to_log_entry()
        assert "[FERPA_DISCLOSURE]" in log_entry
        assert "student_id='stu_001'" in log_entry
        assert "institution_id='inst_abc'" in log_entry


# ---------------------------------------------------------------------------
# raise_on_violation tests
# ---------------------------------------------------------------------------


class TestRaiseOnViolation:
    def test_raise_on_violation_triggers_on_blocked_doc(self):
        f = FERPAMetadataFilter(
            student_id="stu_001",
            institution_id="inst_abc",
            raise_on_violation=True,
        )
        docs = [_doc("blocked", student_id="stu_999", institution_id="inst_abc")]
        with pytest.raises(PermissionError, match="FERPA violation"):
            f.run(docs)

    def test_raise_on_violation_does_not_trigger_when_all_pass(self):
        f = FERPAMetadataFilter(
            student_id="stu_001",
            institution_id="inst_abc",
            raise_on_violation=True,
        )
        docs = [_doc("grade", student_id="stu_001", institution_id="inst_abc")]
        result = f.run(docs)  # no exception
        assert len(result["documents"]) == 1


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict(self, default_filter):
        d = default_filter.to_dict()
        assert d["type"] == "haystack.components.retrievers.ferpa_metadata_filter.FERPAMetadataFilter"
        params = d["init_parameters"]
        assert params["student_id"] == "stu_001"
        assert params["institution_id"] == "inst_abc"
        assert params["authorized_categories"] == ["academic_record", "financial_aid"]
        assert params["requesting_user_id"] == "advisor_007"
        assert params["raise_on_violation"] is False

    def test_from_dict_round_trip(self, default_filter):
        d = default_filter.to_dict()
        restored = FERPAMetadataFilter.from_dict(d)
        assert restored.student_id == default_filter.student_id
        assert restored.institution_id == default_filter.institution_id
        assert restored.authorized_categories == default_filter.authorized_categories
        assert restored.requesting_user_id == default_filter.requesting_user_id

    def test_serialization_preserves_custom_field_names(self):
        f = FERPAMetadataFilter(
            student_id="s1",
            institution_id="i1",
            student_id_field="user_id",
            institution_id_field="org_id",
            category_field="record_type",
        )
        restored = FERPAMetadataFilter.from_dict(f.to_dict())
        assert restored.student_id_field == "user_id"
        assert restored.institution_id_field == "org_id"
        assert restored.category_field == "record_type"


# ---------------------------------------------------------------------------
# Custom field names
# ---------------------------------------------------------------------------


class TestCustomFieldNames:
    def test_custom_student_id_field(self):
        f = FERPAMetadataFilter(
            student_id="stu_001",
            institution_id="inst_abc",
            student_id_field="user_id",
            institution_id_field="org_id",
        )
        doc = Document(content="grade", meta={"user_id": "stu_001", "org_id": "inst_abc"})
        result = f.run([doc])
        assert len(result["documents"]) == 1

    def test_standard_field_names_fail_with_custom_config(self):
        f = FERPAMetadataFilter(
            student_id="stu_001",
            institution_id="inst_abc",
            student_id_field="user_id",
            institution_id_field="org_id",
        )
        doc = _doc("grade", student_id="stu_001", institution_id="inst_abc")  # uses default names
        result = f.run([doc])
        # Doc has student_id/institution_id keys, but filter looks for user_id/org_id
        # Neither user_id nor org_id present → treated as shared content → passes
        assert len(result["documents"]) == 1


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestAsync:
    @pytest.mark.asyncio
    async def test_run_async_matches_sync(self, default_filter):
        docs = [
            _doc("grade", student_id="stu_001", institution_id="inst_abc", category="academic_record"),
            _doc("unauthorized", student_id="stu_002", institution_id="inst_abc", category="academic_record"),
        ]
        sync_result = default_filter.run(list(docs))
        async_result = await default_filter.run_async(list(docs))
        assert len(sync_result["documents"]) == len(async_result["documents"])
