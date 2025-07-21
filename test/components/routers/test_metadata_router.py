# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.components.routers.metadata_router import MetadataRouter
from haystack.dataclasses.byte_stream import ByteStream


class TestMetadataRouter:
    def test_run_document(self):
        rules = {
            "edge_1": {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.created_at", "operator": ">=", "value": "2023-01-01"},
                    {"field": "meta.created_at", "operator": "<", "value": "2023-04-01"},
                ],
            },
            "edge_2": {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.created_at", "operator": ">=", "value": "2023-04-01"},
                    {"field": "meta.created_at", "operator": "<", "value": "2023-07-01"},
                ],
            },
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

    def test_run_bytestream(self):
        rules = {
            "edge_1": {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.created_at", "operator": ">=", "value": "2023-01-01"},
                    {"field": "meta.created_at", "operator": "<", "value": "2023-04-01"},
                ],
            },
            "edge_2": {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.created_at", "operator": ">=", "value": "2023-04-01"},
                    {"field": "meta.created_at", "operator": "<", "value": "2023-07-01"},
                ],
            },
        }
        router = MetadataRouter(rules=rules, output_type=ByteStream)
        bytestreams = [
            ByteStream.from_string("Hello, world!", meta={"created_at": "2023-02-01"}),
            ByteStream.from_string("Hello again!", meta={"created_at": "2023-05-01"}),
            ByteStream.from_string("Goodbye!", meta={"created_at": "2023-08-01"}),
        ]

        output = router.run(documents=bytestreams)
        assert output["edge_1"][0].meta["created_at"] == "2023-02-01"
        assert output["edge_2"][0].meta["created_at"] == "2023-05-01"
        assert output["unmatched"][0].meta["created_at"] == "2023-08-01"

    def test_run_wrong_filter(self):
        rules = {
            "edge_1": {"field": "meta.created_at", "operator": ">=", "value": "2023-01-01"},
            "wrong_filter": {"wrong_value": "meta.created_at == 2023-04-01"},
        }
        with pytest.raises(ValueError):
            MetadataRouter(rules=rules)

    def test_run_datetime_with_timezone(self):
        rules = {
            "edge_1": {
                "operator": "AND",
                "conditions": [{"field": "meta.created_at", "operator": ">=", "value": "2025-02-01"}],
            }
        }
        router = MetadataRouter(rules=rules)
        documents = [
            Document(meta={"created_at": "2025-02-03T12:45:46.435816Z"}),
            Document(meta={"created_at": "2025-02-01T12:45:46.435816Z"}),
            Document(meta={"created_at": "2025-01-03T12:45:46.435816Z"}),
        ]
        output = router.run(documents=documents)
        assert len(output["edge_1"]) == 2
        assert output["edge_1"][0].meta["created_at"] == "2025-02-03T12:45:46.435816Z"
        assert output["edge_1"][1].meta["created_at"] == "2025-02-01T12:45:46.435816Z"
        assert output["unmatched"][0].meta["created_at"] == "2025-01-03T12:45:46.435816Z"
