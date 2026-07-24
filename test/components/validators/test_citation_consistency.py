# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.components.validators import Citation, CitationConsistencyChecker
from haystack.dataclasses import Document


@pytest.fixture
def documents():
    return [
        Document(
            id="veltranib-rct",
            content=(
                "In the veltranib randomized controlled trial, fasting plasma glucose fell by 28 mg/dL in the "
                "veltranib arm. Body weight was unchanged in both arms."
            ),
        ),
        Document(
            id="restatin-meta",
            content=(
                "Restatin reduced the relative risk of non-fatal myocardial infarction by 25%. In the low-risk "
                "subgroup the reduction was not statistically significant."
            ),
        ),
    ]


class TestCitationConsistencyChecker:
    def test_real_quote_is_consistent(self, documents):
        checker = CitationConsistencyChecker()
        result = checker.run(
            citations=[Citation("Weight did not change.", "veltranib-rct", "Body weight was unchanged in both arms")],
            documents=documents,
        )
        assert len(result["consistent"]) == 1
        assert result["consistent"][0].status == "found"
        assert result["inconsistent"] == []

    def test_typography_and_case_insensitive(self, documents):
        checker = CitationConsistencyChecker()
        result = checker.run(
            citations=[Citation("c", "veltranib-rct", "  BODY   weight was unchanged in both arms.  ")],
            documents=documents,
        )
        assert result["consistent"][0].status == "found"

    def test_frankenquote_is_inconsistent(self, documents):
        # every word is real, but this sentence was never written in restatin-meta
        franken = (
            "Restatin reduced the relative risk of non-fatal myocardial infarction by 25% in the low-risk subgroup."
        )
        checker = CitationConsistencyChecker()
        result = checker.run(citations=[Citation("c", "restatin-meta", franken)], documents=documents)
        assert result["consistent"] == []
        assert result["inconsistent"][0].status == "not_found"

    def test_fabrication_is_inconsistent(self, documents):
        checker = CitationConsistencyChecker()
        result = checker.run(
            citations=[Citation("c", "veltranib-rct", "Veltranib cured the disease in every patient")],
            documents=documents,
        )
        assert result["inconsistent"][0].status == "not_found"

    def test_misattributed_quote_is_inconsistent_by_default(self, documents):
        # a real quote from veltranib-rct, but attributed to restatin-meta
        checker = CitationConsistencyChecker()
        result = checker.run(
            citations=[Citation("c", "restatin-meta", "Body weight was unchanged in both arms")], documents=documents
        )
        assert result["consistent"] == []
        assert result["inconsistent"][0].status == "misattributed"

    def test_misattributed_routed_to_consistent_when_configured(self, documents):
        checker = CitationConsistencyChecker(treat_misattributed_as_consistent=True)
        result = checker.run(
            citations=[Citation("c", "restatin-meta", "Body weight was unchanged in both arms")], documents=documents
        )
        assert result["consistent"][0].status == "misattributed"
        assert result["inconsistent"] == []

    def test_empty_quote_fails_closed(self, documents):
        checker = CitationConsistencyChecker()
        result = checker.run(citations=[Citation("c", "veltranib-rct", "   ")], documents=documents)
        assert result["inconsistent"][0].status == "not_found"

    def test_unknown_document_id_does_not_raise(self, documents):
        checker = CitationConsistencyChecker()
        result = checker.run(
            citations=[Citation("c", "does-not-exist", "Body weight was unchanged in both arms")], documents=documents
        )
        # real quote, but cited doc id is unknown -> found in another doc -> misattributed
        assert result["inconsistent"][0].status == "misattributed"

    def test_documents_without_content_are_ignored(self):
        checker = CitationConsistencyChecker()
        result = checker.run(citations=[Citation("c", "d1", "anything")], documents=[Document(id="d1", content=None)])
        assert result["inconsistent"][0].status == "not_found"

    def test_mixed_batch_routing(self, documents):
        checker = CitationConsistencyChecker()
        result = checker.run(
            citations=[
                Citation("c1", "veltranib-rct", "Body weight was unchanged in both arms"),  # found
                Citation("c2", "veltranib-rct", "totally made up quote"),  # not_found
            ],
            documents=documents,
        )
        assert len(result["consistent"]) == 1
        assert len(result["inconsistent"]) == 1

    def test_to_dict(self):
        checker = CitationConsistencyChecker(treat_misattributed_as_consistent=True)
        data = checker.to_dict()
        assert data == {
            "type": "haystack.components.validators.citation_consistency.CitationConsistencyChecker",
            "init_parameters": {"treat_misattributed_as_consistent": True},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.validators.citation_consistency.CitationConsistencyChecker",
            "init_parameters": {"treat_misattributed_as_consistent": True},
        }
        checker = CitationConsistencyChecker.from_dict(data)
        assert checker.treat_misattributed_as_consistent is True

    def test_serialization_roundtrip_in_pipeline(self):
        pipe = Pipeline()
        pipe.add_component("checker", CitationConsistencyChecker())
        pipe_yaml = pipe.dumps()
        new_pipe = Pipeline.loads(pipe_yaml)
        assert new_pipe.get_component("checker").treat_misattributed_as_consistent is False
