import pytest

from haystack.preview import Document
from haystack.preview.components.routers.metadata_router import MetadataRouter


class TestMetadataRouter:
    @pytest.mark.unit
    def test_run(self):
        rules = {
            "edge_1": {"created_at": {"$gte": "2023-01-01", "$lt": "2023-04-01"}},
            "edge_2": {"created_at": {"$gte": "2023-04-01", "$lt": "2023-07-01"}},
        }
        router = MetadataRouter(rules=rules)
        documents = [
            Document(meta={"created_at": "2023-02-01"}),
            Document(meta={"created_at": "2023-05-01"}),
            Document(meta={"created_at": "2023-08-01"}),
        ]
        output = router.run(documents=documents)
        assert output["edge_1"][0].meta["created_at"] == "2023-02-01"
        assert output["edge_2"][0].meta["created_at"] == "2023-05-01"
        assert output["unmatched"][0].meta["created_at"] == "2023-08-01"
