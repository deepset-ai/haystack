# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.routers import DocumentLengthRouter
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import Document


class TestDocumentLengthRouter:
    def test_init(self):
        router = DocumentLengthRouter(threshold=20)
        assert router.threshold == 20

    def test_run(self):
        docs = [Document(content="Short"), Document(content="Long document " * 20)]
        router = DocumentLengthRouter(threshold=10)
        result = router.run(documents=docs)

        assert len(result["short_documents"]) == 1
        assert len(result["long_documents"]) == 1
        assert result["short_documents"][0] == docs[0]
        assert result["long_documents"][0] == docs[1]

    def test_run_with_null_content(self):
        docs = [Document(content=None), Document(content="Long document " * 20)]
        router = DocumentLengthRouter(threshold=10)
        result = router.run(documents=docs)

        assert len(result["short_documents"]) == 1
        assert len(result["long_documents"]) == 1
        assert result["short_documents"][0] == docs[0]
        assert result["long_documents"][0] == docs[1]

    def test_run_with_negative_threshold(self):
        docs = [Document(content=None), Document(content="Short"), Document(content="Long document " * 20)]
        router = DocumentLengthRouter(threshold=-1)
        result = router.run(documents=docs)

        assert len(result["short_documents"]) == 1
        assert len(result["long_documents"]) == 2
        assert result["short_documents"][0] == docs[0]
        assert result["long_documents"][0] == docs[1]
        assert result["long_documents"][1] == docs[2]

    def test_to_dict(self):
        router = DocumentLengthRouter(threshold=10)
        expected_dict = {
            "type": "haystack.components.routers.document_length_router.DocumentLengthRouter",
            "init_parameters": {"threshold": 10},
        }
        assert component_to_dict(router, "router") == expected_dict

    def test_from_dict(self):
        router_dict = {
            "type": "haystack.components.routers.document_length_router.DocumentLengthRouter",
            "init_parameters": {"threshold": 10},
        }
        loaded_router = component_from_dict(DocumentLengthRouter, router_dict, name="router")

        assert isinstance(loaded_router, DocumentLengthRouter)
        assert loaded_router.threshold == 10
