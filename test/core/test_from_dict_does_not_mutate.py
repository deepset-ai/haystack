# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""`from_dict` must not write deserialized objects back into the caller's mapping.

`default_from_dict` copies `init_parameters` before replacing serialized
sub-objects (see the comment there), precisely so that a second deserialization
does not receive already-parsed objects. Several components did their own
deserialization into `data["init_parameters"]` *before* calling it, which
defeats that copy: the caller's dict ends up holding live objects, so it can no
longer be JSON-serialized and a second `from_dict` fails.

Same defect previously fixed one component at a time: MetadataRouter (#13421),
MockChatGenerator (#13471), ListJoiner (#13563), Tool (#13606). This test pins
the invariant for the components fixed alongside it.
"""

import json
from copy import deepcopy

import pytest

from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy, FilterPolicy

CASES = [
    pytest.param(
        DocumentWriter,
        lambda: DocumentWriter(document_store=InMemoryDocumentStore(), policy=DuplicatePolicy.OVERWRITE),
        id="document-writer-enum",
    ),
    pytest.param(
        InMemoryBM25Retriever,
        lambda: InMemoryBM25Retriever(document_store=InMemoryDocumentStore(), filter_policy=FilterPolicy.MERGE),
        id="bm25-retriever-enum",
    ),
    pytest.param(
        InMemoryEmbeddingRetriever,
        lambda: InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore(), filter_policy=FilterPolicy.MERGE),
        id="embedding-retriever-enum",
    ),
    pytest.param(
        ChatPromptBuilder,
        lambda: ChatPromptBuilder(template=[ChatMessage.from_user("Hello")]),
        id="chat-prompt-builder-nested-message",
    ),
]


@pytest.mark.parametrize("cls, build", CASES)
class TestFromDictDoesNotMutateCallerDict:
    def test_caller_dict_is_unchanged(self, cls, build):
        payload = build().to_dict()
        snapshot = deepcopy(payload)

        cls.from_dict(payload)

        assert payload == snapshot

    def test_payload_stays_json_serializable(self, cls, build):
        payload = build().to_dict()

        cls.from_dict(payload)

        json.dumps(payload)

    def test_second_deserialization_succeeds(self, cls, build):
        payload = build().to_dict()

        first = cls.from_dict(payload)
        second = cls.from_dict(payload)

        assert type(first) is type(second)
