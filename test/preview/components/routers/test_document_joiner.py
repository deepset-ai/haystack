import logging

import pytest

from haystack.preview import Document
from haystack.preview.components.routers.document_joiner import DocumentJoiner


class TestDocumentJoiner:
    @pytest.mark.unit
    def test_init(self):
        joiner = DocumentJoiner()
        assert joiner.join_mode == "concatenate"
        assert joiner.weights is None
        assert joiner.top_k is None
        assert joiner.sort_by_score

    @pytest.mark.unit
    def test_init_with_custom_parameters(self):
        joiner = DocumentJoiner(join_mode="merge", weights=[0.4, 0.6], top_k=5, sort_by_score=False)
        assert joiner.join_mode == "merge"
        assert joiner.weights == [0.4, 0.6]
        assert joiner.top_k == 5
        assert not joiner.sort_by_score

    @pytest.mark.unit
    def test_empty_list(self):
        joiner = DocumentJoiner()
        result = joiner.run([])
        assert result == {"documents": []}

    @pytest.mark.unit
    def test_list_of_empty_lists(self):
        joiner = DocumentJoiner()
        result = joiner.run([[], []])
        assert result == {"documents": []}

    @pytest.mark.unit
    def test_list_with_one_empty_list(self):
        joiner = DocumentJoiner()
        documents = [Document(text="a"), Document(text="b"), Document(text="c")]
        result = joiner.run([[], documents])
        assert result == {"documents": documents}

    @pytest.mark.unit
    def test_unsupported_join_mode(self):
        with pytest.raises(ValueError, match="DocumentJoiner component does not support 'unsupported_mode' join_mode."):
            DocumentJoiner(join_mode="unsupported_mode")

    @pytest.mark.unit
    def test_run_with_concatenate_join_mode_and_top_k(self):
        joiner = DocumentJoiner(top_k=6)
        documents_1 = [Document(text="a"), Document(text="b"), Document(text="c")]
        documents_2 = [
            Document(text="d"),
            Document(text="e"),
            Document(text="f", metadata={"key": "value"}),
            Document(text="g"),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 6
        assert sorted(documents_1 + documents_2[:-1], key=lambda d: d.id) == sorted(
            output["documents"], key=lambda d: d.id
        )

    @pytest.mark.unit
    def test_run_with_concatenate_join_mode_and_duplicate_documents(self):
        joiner = DocumentJoiner()
        documents_1 = [Document(text="a", score=0.3, metadata={"key": "1"}), Document(text="b"), Document(text="c")]
        documents_2 = [
            Document(text="a", score=0.2, metadata={"key": "2"}),
            Document(text="a"),
            Document(text="f", metadata={"key": "value"}),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 4
        assert sorted(documents_1 + [documents_2[-1]], key=lambda d: d.id) == sorted(
            output["documents"], key=lambda d: d.id
        )

    @pytest.mark.unit
    def test_run_with_merge_join_mode(self):
        joiner = DocumentJoiner(join_mode="merge", weights=[1.5, 0.5])
        documents_1 = [Document(text="a", score=1.0), Document(text="b", score=2.0)]
        documents_2 = [
            Document(text="a", score=0.5),
            Document(text="b", score=3.0),
            Document(text="f", score=4.0, metadata={"key": "value"}),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 3
        expected_documents = [
            Document(text="a", score=1.25),
            Document(text="b", score=2.25),
            Document(text="f", score=4.0, metadata={"key": "value"}),
        ]
        assert sorted(expected_documents, key=lambda d: d.id) == sorted(output["documents"], key=lambda d: d.id)

    @pytest.mark.unit
    def test_run_with_reciprocal_rank_fusion_join_mode(self):
        joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")
        documents_1 = [Document(text="a"), Document(text="b"), Document(text="c")]
        documents_2 = [
            Document(text="b", score=1000.0),
            Document(text="c"),
            Document(text="a"),
            Document(text="f", metadata={"key": "value"}),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 4
        expected_documents = [
            Document(text="b"),
            Document(text="a"),
            Document(text="c"),
            Document(text="f", metadata={"key": "value"}),
        ]
        assert expected_documents == output["documents"]

    @pytest.mark.unit
    def test_sort_by_score_without_scores(self, caplog):
        joiner = DocumentJoiner()
        with caplog.at_level(logging.INFO):
            documents = [Document(text="a"), Document(text="b", score=0.5)]
            output = joiner.run([documents])
            assert "documents were sorted as if their score was `-infinity`" in caplog.text
            assert output["documents"] == documents[::-1]

    @pytest.mark.unit
    def test_output_documents_not_sorted_by_score(self):
        joiner = DocumentJoiner(sort_by_score=False)
        documents_1 = [Document(text="a", score=0.1)]
        documents_2 = [Document(text="d", score=0.2)]
        output = joiner.run([documents_1, documents_2])
        assert output["documents"] == documents_1 + documents_2
