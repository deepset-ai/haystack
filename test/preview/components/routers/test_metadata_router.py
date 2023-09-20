import pytest

from haystack.preview import Document
from haystack.preview.components.routers.metadata_router import MetadataRouter


class TestMetadataRouter:
    @pytest.mark.unit
    def test_to_dict(self):
        component = MetadataRouter(rules={"edge_1": {"created_at": {"$gte": "2023-01-01", "$lt": "2023-04-01"}}})
        data = component.to_dict()
        assert data == {
            "type": "MetadataRouter",
            "init_parameters": {"rules": {"edge_1": {"created_at": {"$gte": "2023-01-01", "$lt": "2023-04-01"}}}},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "MetadataRouter",
            "init_parameters": {"rules": {"edge_1": {"created_at": {"$gte": "2023-01-01", "$lt": "2023-04-01"}}}},
        }
        component = MetadataRouter.from_dict(data)
        assert component.rules == {"edge_1": {"created_at": {"$gte": "2023-01-01", "$lt": "2023-04-01"}}}

    @pytest.mark.unit
    def test_run(self):
        rules = {
            "edge_1": {"created_at": {"$gte": "2023-01-01", "$lt": "2023-04-01"}},
            "edge_2": {"created_at": {"$gte": "2023-04-01", "$lt": "2023-07-01"}},
        }
        router = MetadataRouter(rules=rules)
        documents = [
            Document(metadata={"created_at": "2023-02-01"}),
            Document(metadata={"created_at": "2023-05-01"}),
            Document(metadata={"created_at": "2023-08-01"}),
        ]
        output = router.run(documents=documents)
        assert output["edge_1"][0].metadata["created_at"] == "2023-02-01"
        assert output["edge_2"][0].metadata["created_at"] == "2023-05-01"
        assert output["unmatched"][0].metadata["created_at"] == "2023-08-01"
