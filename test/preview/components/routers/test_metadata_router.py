import pytest

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
