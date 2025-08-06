# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import pytest

from haystack.components.routers.metadata_router import MetadataRouter
from haystack.dataclasses import ByteStream, Document


class TestMetadataRouter:
    def test_run(self):
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

    def test_run_with_byte_stream(self):
        byt1 = ByteStream.from_string(text="What is this", meta={"language": "en"})
        byt2 = ByteStream.from_string(text="Berlin ist die Haupststadt von Deutschland.", meta={"language": "de"})
        docs = [byt1, byt2]
        router = MetadataRouter(
            rules={"en": {"field": "meta.language", "operator": "==", "value": "en"}}, output_type=List[ByteStream]
        )
        output = router.run(documents=docs)
        assert output["en"][0].data == byt1.data
        assert output["unmatched"][0].data == byt2.data

    def test_run_with_mixed_documents_and_byte_streams(self):
        byt1 = ByteStream.from_string(text="What is this", meta={"language": "en"})
        byt2 = ByteStream.from_string(text="Berlin ist die Haupststadt von Deutschland.", meta={"language": "de"})
        doc1 = Document(content="What is this", meta={"language": "en"})
        doc2 = Document(content="Berlin ist die Haupststadt von Deutschland.", meta={"language": "de"})
        docs = [byt1, byt2, doc1, doc2]
        router = MetadataRouter(
            rules={"en": {"field": "meta.language", "operator": "==", "value": "en"}},
            output_type=List[Union[Document, ByteStream]],
        )
        output = router.run(documents=docs)
        assert output["en"][0].data == byt1.data
        assert output["en"][1].content == "What is this"
        assert output["unmatched"][0].data == byt2.data
        assert output["unmatched"][1].content == "Berlin ist die Haupststadt von Deutschland."

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

    def test_to_dict(self):
        rules = {
            "edge_1": {
                "operator": "AND",
                "conditions": [{"field": "meta.created_at", "operator": ">=", "value": "2025-02-01"}],
            }
        }
        router = MetadataRouter(rules=rules)
        expected_dict = {
            "type": "haystack.components.routers.metadata_router.MetadataRouter",
            "init_parameters": {"rules": rules, "output_type": "typing.List[haystack.dataclasses.document.Document]"},
        }
        assert router.to_dict() == expected_dict

    def test_to_dict_with_parameters(self):
        rules = {
            "edge_1": {
                "operator": "AND",
                "conditions": [{"field": "meta.created_at", "operator": ">=", "value": "2025-02-01"}],
            }
        }
        router = MetadataRouter(rules=rules, output_type=List[Union[ByteStream, Document]])
        expected_dict = {
            "type": "haystack.components.routers.metadata_router.MetadataRouter",
            "init_parameters": {
                "rules": rules,
                "output_type": """typing.List[typing.Union[haystack.dataclasses.document.Document,
                haystack.dataclasses.byte_stream.ByteStream]]""",
            },
        }
        assert router.to_dict() == expected_dict

    def test_from_dict(self):
        router_dict = {
            "type": "haystack.components.routers.metadata_router.MetadataRouter",
            "init_parameters": {
                "rules": {
                    "edge_1": {
                        "operator": "AND",
                        "conditions": [{"field": "meta.created_at", "operator": ">=", "value": "2025-02-01"}],
                    }
                },
                "output_type": "typing.List[haystack.dataclasses.document.Document]",
            },
        }
        router = MetadataRouter.from_dict(router_dict)
        assert router.rules == {
            "edge_1": {
                "operator": "AND",
                "conditions": [{"field": "meta.created_at", "operator": ">=", "value": "2025-02-01"}],
            }
        }
        assert router.output_type == List[Document]

    def test_from_dict_with_parameters(self):
        router_dict = {
            "type": "haystack.components.routers.metadata_router.MetadataRouter",
            "init_parameters": {
                "rules": {
                    "edge_1": {
                        "operator": "AND",
                        "conditions": [{"field": "meta.created_at", "operator": ">=", "value": "2025-02-01"}],
                    }
                },
                "output_type": """typing.List[typing.Union[haystack.dataclasses.document.Document,
                haystack.dataclasses.byte_stream.ByteStream]]""",
            },
        }
        router = MetadataRouter.from_dict(router_dict)
        assert router.rules == {
            "edge_1": {
                "operator": "AND",
                "conditions": [{"field": "meta.created_at", "operator": ">=", "value": "2025-02-01"}],
            }
        }
        assert router.output_type == List[Union[ByteStream, Document]]
