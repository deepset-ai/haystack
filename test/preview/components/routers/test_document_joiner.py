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
    def test_to_dict(self):
        joiner = DocumentJoiner()
        data = joiner.to_dict()
        assert data == {
            "type": "DocumentJoiner",
            "init_parameters": {"join_mode": "concatenate", "weights": None, "top_k": None, "sort_by_score": True},
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        joiner = DocumentJoiner(join_mode="merge", weights=[0.4, 0.6], top_k=5, sort_by_score=False)
        data = joiner.to_dict()
        assert data == {
            "type": "DocumentJoiner",
            "init_parameters": {"join_mode": "merge", "weights": [0.4, 0.6], "top_k": 5, "sort_by_score": False},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "DocumentJoiner",
            "init_parameters": {"join_mode": "merge", "weights": [0.4, 0.6], "top_k": 5, "sort_by_score": False},
        }
        joiner = DocumentJoiner.from_dict(data)
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
        result = joiner.run([], [])
        assert result == {"documents": []}

    @pytest.mark.unit
    def test_list_with_one_empty_list(self):
        joiner = DocumentJoiner()
        documents_1 = [
            Document(metadata={"created_at": "2023-02-01"}),
            Document(metadata={"created_at": "2023-05-01"}),
            Document(metadata={"created_at": "2023-08-01"}),
        ]
        result = joiner.run([], documents_1)
        assert result == {"documents": documents_1}

    @pytest.mark.unit
    def test_unsupported_join_mode(self):
        with pytest.raises(ValueError, match="DocumentJoiner component does not support 'unsupported_mode' join_mode."):
            DocumentJoiner(join_mode="unsupported_mode")

    @pytest.mark.unit
    def test_run(self):
        joiner = DocumentJoiner()
        documents_1 = [
            Document(metadata={"created_at": "2023-02-01"}),
            Document(metadata={"created_at": "2023-05-01"}),
            Document(metadata={"created_at": "2023-08-01"}),
        ]
        documents_2 = [
            Document(metadata={"created_at": "2023-02-01"}),
            Document(metadata={"created_at": "2023-05-01"}),
            Document(metadata={"created_at": "2023-08-01"}),
        ]
        output = joiner.run(documents_1, documents_2)
        assert len(output["documents"]) == 6
        assert sorted(documents_1 + documents_2, key=lambda d: d.id) == sorted(output["documents"], key=lambda d: d.id)
