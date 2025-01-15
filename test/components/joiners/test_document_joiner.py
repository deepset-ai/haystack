# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import re

import pytest

from haystack import Document
from haystack.components.joiners.document_joiner import DocumentJoiner, JoinMode


class TestDocumentJoiner:
    def test_init(self):
        joiner = DocumentJoiner()
        assert joiner.join_mode == JoinMode.CONCATENATE
        assert joiner.weights is None
        assert joiner.top_k is None
        assert joiner.sort_by_score

    def test_init_with_custom_parameters(self):
        joiner = DocumentJoiner(join_mode="merge", weights=[0.4, 0.6], top_k=5, sort_by_score=False)
        assert joiner.join_mode == JoinMode.MERGE
        assert joiner.weights == [0.4, 0.6]
        assert joiner.top_k == 5
        assert not joiner.sort_by_score

    def test_to_dict(self):
        joiner = DocumentJoiner()
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.document_joiner.DocumentJoiner",
            "init_parameters": {"join_mode": "concatenate", "sort_by_score": True, "top_k": None, "weights": None},
        }

    def test_to_dict_custom_parameters(self):
        joiner = DocumentJoiner("merge", weights=[0.4, 0.6], top_k=4, sort_by_score=False)
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.document_joiner.DocumentJoiner",
            "init_parameters": {"join_mode": "merge", "weights": [0.4, 0.6], "top_k": 4, "sort_by_score": False},
        }

    def test_from_dict(self):
        data = {"type": "haystack.components.joiners.document_joiner.DocumentJoiner", "init_parameters": {}}
        document_joiner = DocumentJoiner.from_dict(data)
        assert document_joiner.join_mode == JoinMode.CONCATENATE
        assert document_joiner.weights == None
        assert document_joiner.top_k == None
        assert document_joiner.sort_by_score

    def test_from_dict_customs_parameters(self):
        data = {
            "type": "haystack.components.joiners.document_joiner.DocumentJoiner",
            "init_parameters": {"join_mode": "merge", "weights": [0.5, 0.6], "top_k": 6, "sort_by_score": False},
        }
        document_joiner = DocumentJoiner.from_dict(data)
        assert document_joiner.join_mode == JoinMode.MERGE
        assert document_joiner.weights == pytest.approx([0.5, 0.6], rel=0.1)
        assert document_joiner.top_k == 6
        assert not document_joiner.sort_by_score

    @pytest.mark.parametrize(
        "join_mode",
        [
            JoinMode.CONCATENATE,
            JoinMode.MERGE,
            JoinMode.RECIPROCAL_RANK_FUSION,
            JoinMode.DISTRIBUTION_BASED_RANK_FUSION,
        ],
    )
    def test_empty_list(self, join_mode: JoinMode):
        joiner = DocumentJoiner(join_mode=join_mode)
        result = joiner.run([])
        assert result == {"documents": []}

    @pytest.mark.parametrize(
        "join_mode",
        [
            JoinMode.CONCATENATE,
            JoinMode.MERGE,
            JoinMode.RECIPROCAL_RANK_FUSION,
            JoinMode.DISTRIBUTION_BASED_RANK_FUSION,
        ],
    )
    def test_list_of_empty_lists(self, join_mode: JoinMode):
        joiner = DocumentJoiner(join_mode=join_mode)
        result = joiner.run([[], []])
        assert result == {"documents": []}

    @pytest.mark.parametrize(
        "join_mode",
        [
            JoinMode.CONCATENATE,
            JoinMode.MERGE,
            JoinMode.RECIPROCAL_RANK_FUSION,
            JoinMode.DISTRIBUTION_BASED_RANK_FUSION,
        ],
    )
    def test_list_with_one_empty_list(self, join_mode: JoinMode):
        joiner = DocumentJoiner(join_mode=join_mode)
        documents = [Document(content="a"), Document(content="b"), Document(content="c")]
        result = joiner.run([[], documents])
        assert result == {"documents": documents}

    def test_unsupported_join_mode(self):
        unsupported_mode = "unsupported_mode"
        expected_error_pattern = (
            re.escape(f"Unknown join mode '{unsupported_mode}'") + r".*Supported modes in DocumentJoiner are: \[.*\]"
        )

        with pytest.raises(ValueError, match=expected_error_pattern):
            DocumentJoiner(join_mode=unsupported_mode)

    def test_run_with_concatenate_join_mode_and_top_k(self):
        joiner = DocumentJoiner(top_k=6)
        documents_1 = [Document(content="a"), Document(content="b"), Document(content="c")]
        documents_2 = [
            Document(content="d"),
            Document(content="e"),
            Document(content="f", meta={"key": "value"}),
            Document(content="g"),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 6
        assert sorted(documents_1 + documents_2[:-1], key=lambda d: d.id) == sorted(
            output["documents"], key=lambda d: d.id
        )

    def test_run_with_concatenate_join_mode_and_duplicate_documents(self):
        joiner = DocumentJoiner()
        documents_1 = [Document(content="a", score=0.3), Document(content="b"), Document(content="c")]
        documents_2 = [
            Document(content="a", score=0.2),
            Document(content="a"),
            Document(content="f", meta={"key": "value"}),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 4
        assert sorted(documents_1 + [documents_2[-1]], key=lambda d: d.id) == sorted(
            output["documents"], key=lambda d: d.id
        )

    def test_run_with_merge_join_mode(self):
        joiner = DocumentJoiner(join_mode="merge", weights=[1.5, 0.5])
        documents_1 = [Document(content="a", score=1.0), Document(content="b", score=2.0)]
        documents_2 = [
            Document(content="a", score=0.5),
            Document(content="b", score=3.0),
            Document(content="f", score=4.0, meta={"key": "value"}),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 3
        expected_document_ids = [
            doc.id
            for doc in [
                Document(content="a", score=1.25),
                Document(content="b", score=2.25),
                Document(content="f", score=4.0, meta={"key": "value"}),
            ]
        ]
        assert all(doc.id in expected_document_ids for doc in output["documents"])

    def test_run_with_reciprocal_rank_fusion_join_mode(self):
        joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")
        documents_1 = [Document(content="a"), Document(content="b"), Document(content="c")]
        documents_2 = [
            Document(content="b", score=1000.0),
            Document(content="c"),
            Document(content="a"),
            Document(content="f", meta={"key": "value"}),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 4
        expected_document_ids = [
            doc.id
            for doc in [
                Document(content="b"),
                Document(content="a"),
                Document(content="c"),
                Document(content="f", meta={"key": "value"}),
            ]
        ]
        assert all(doc.id in expected_document_ids for doc in output["documents"])

    def test_run_with_distribution_based_rank_fusion_join_mode(self):
        joiner = DocumentJoiner(join_mode="distribution_based_rank_fusion")
        documents_1 = [
            Document(content="a", score=0.6),
            Document(content="b", score=0.2),
            Document(content="c", score=0.5),
        ]
        documents_2 = [
            Document(content="d", score=0.5),
            Document(content="e", score=0.8),
            Document(content="f", score=1.1, meta={"key": "value"}),
            Document(content="g", score=0.3),
            Document(content="a", score=0.3),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 7
        expected_document_ids = [
            doc.id
            for doc in [
                Document(content="a", score=0.66),
                Document(content="b", score=0.27),
                Document(content="c", score=0.56),
                Document(content="d", score=0.44),
                Document(content="e", score=0.60),
                Document(content="f", score=0.76, meta={"key": "value"}),
                Document(content="g", score=0.33),
            ]
        ]
        assert all(doc.id in expected_document_ids for doc in output["documents"])

    def test_run_with_distribution_based_rank_fusion_join_mode_same_scores(self):
        joiner = DocumentJoiner(join_mode="distribution_based_rank_fusion")
        documents_1 = [
            Document(content="a", score=0.2),
            Document(content="b", score=0.2),
            Document(content="c", score=0.2),
        ]
        documents_2 = [
            Document(content="d", score=0.5),
            Document(content="e", score=0.8),
            Document(content="f", score=1.1, meta={"key": "value"}),
            Document(content="g", score=0.3),
            Document(content="a", score=0.3),
        ]
        output = joiner.run([documents_1, documents_2])
        assert len(output["documents"]) == 7
        expected_document_ids = [
            doc.id
            for doc in [
                Document(content="a", score=0),
                Document(content="b", score=0),
                Document(content="c", score=0),
                Document(content="d", score=0.44),
                Document(content="e", score=0.60),
                Document(content="f", score=0.76, meta={"key": "value"}),
                Document(content="g", score=0.33),
            ]
        ]
        assert all(doc.id in expected_document_ids for doc in output["documents"])

    def test_run_with_top_k_in_run_method(self):
        joiner = DocumentJoiner()
        documents_1 = [Document(content="a"), Document(content="b"), Document(content="c")]
        documents_2 = [Document(content="d"), Document(content="e"), Document(content="f")]
        top_k = 4
        output = joiner.run([documents_1, documents_2], top_k=top_k)
        assert len(output["documents"]) == top_k

    def test_sort_by_score_without_scores(self, caplog):
        joiner = DocumentJoiner()
        with caplog.at_level(logging.INFO):
            documents = [Document(content="a"), Document(content="b", score=0.5)]
            output = joiner.run([documents])
            assert "those with score=None were sorted as if they had a score of -infinity" in caplog.text
            assert output["documents"] == documents[::-1]

    def test_output_documents_not_sorted_by_score(self):
        joiner = DocumentJoiner(sort_by_score=False)
        documents_1 = [Document(content="a", score=0.1)]
        documents_2 = [Document(content="d", score=0.2)]
        output = joiner.run([documents_1, documents_2])
        assert output["documents"] == documents_1 + documents_2

    def test_test_score_norm_with_rrf(self):
        """
        Verifies reciprocal rank fusion (RRF) of the DocumentJoiner component with various weight configurations.
        It creates a set of documents, forms them into two lists, and then applies multiple DocumentJoiner
        instances with distinct weights to these lists. The test checks if the resulting
        joined documents are correctly sorted in descending order by score, ensuring the RRF ranking works as
        expected under different weighting scenarios.
        """
        num_docs = 6
        docs = []

        for i in range(num_docs):
            docs.append(Document(content=f"doc{i}"))

        docs_2 = [docs[0], docs[4], docs[2], docs[5], docs[1]]
        document_lists = [docs, docs_2]

        joiner_1 = DocumentJoiner(join_mode="reciprocal_rank_fusion", weights=[0.5, 0.5])

        joiner_2 = DocumentJoiner(join_mode="reciprocal_rank_fusion", weights=[7, 7])

        joiner_3 = DocumentJoiner(join_mode="reciprocal_rank_fusion", weights=[0.7, 0.3])

        joiner_4 = DocumentJoiner(join_mode="reciprocal_rank_fusion", weights=[0.6, 0.4])

        joiner_5 = DocumentJoiner(join_mode="reciprocal_rank_fusion", weights=[1, 0])

        joiners = [joiner_1, joiner_2, joiner_3, joiner_4, joiner_5]

        for index, joiner in enumerate(joiners):
            join_results = joiner.run(documents=document_lists)
            is_sorted = all(
                join_results["documents"][i].score >= join_results["documents"][i + 1].score
                for i in range(len(join_results["documents"]) - 1)
            )

            assert (
                is_sorted
            ), "Documents are not sorted in descending order by score, there is an issue with rff ranking"
