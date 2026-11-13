# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from haystack import Document, Pipeline
from haystack.components.preprocessors.semantic_resolver import SemanticResolver


class TestSemanticResolver:
    def test_init_defaults(self):
        resolver = SemanticResolver()
        assert resolver.url == "http://localhost:8080/similarity"
        assert resolver.threshold == 0.8
        assert resolver.metadata_key == "entities"
        assert resolver.entity_key == "entity"
        assert resolver.type_key == "entity_type"
        assert resolver.group_by_type is True
        assert resolver.canonical_name_strategy == "most_frequent"
        assert resolver.keep_original_entity_name is True
        assert resolver.deduplicate_entities is True
        assert resolver.batch_size == 64
        assert resolver.timeout == 30.0

    def test_init_custom_parameters(self):
        resolver = SemanticResolver(
            url="https://tei.example.com/similarity/",
            threshold=0.9,
            metadata_key="extracted_entities",
            entity_key="name",
            type_key="category",
            group_by_type=False,
            canonical_name_strategy="shortest",
            keep_original_entity_name=False,
            deduplicate_entities=False,
            batch_size=32,
            timeout=15.0,
            http_client_kwargs={"headers": {"X-Custom": "test"}},
        )
        assert resolver.url == "https://tei.example.com/similarity"
        assert resolver.threshold == 0.9
        assert resolver.metadata_key == "extracted_entities"
        assert resolver.entity_key == "name"
        assert resolver.type_key == "category"
        assert resolver.group_by_type is False
        assert resolver.canonical_name_strategy == "shortest"
        assert resolver.keep_original_entity_name is False
        assert resolver.deduplicate_entities is False
        assert resolver.batch_size == 32
        assert resolver.timeout == 15.0
        assert resolver.http_client_kwargs == {"headers": {"X-Custom": "test"}}

    def test_init_rejects_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            SemanticResolver(threshold=-0.1)

        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            SemanticResolver(threshold=1.1)

    def test_init_rejects_invalid_batch_size_or_timeout(self):
        with pytest.raises(ValueError, match="batch_size must be greater than 0"):
            SemanticResolver(batch_size=0)

        with pytest.raises(ValueError, match="timeout must be greater than 0"):
            SemanticResolver(timeout=0)

    def test_init_rejects_invalid_canonical_strategy(self):
        with pytest.raises(ValueError, match="canonical_name_strategy must be one of"):
            SemanticResolver(canonical_name_strategy="unsupported")  # type: ignore[arg-type]

    def test_init_rejects_empty_keys(self):
        with pytest.raises(ValueError, match="metadata_key cannot be empty"):
            SemanticResolver(metadata_key="")

        with pytest.raises(ValueError, match="entity_key cannot be empty"):
            SemanticResolver(entity_key="")

    def test_to_dict_and_from_dict(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "fake_hf_token")
        resolver = SemanticResolver(
            url="http://custom-tei:8080",
            threshold=0.85,
            metadata_key="ents",
            canonical_name_strategy="longest",
            batch_size=16,
        )
        serialized = resolver.to_dict()
        assert serialized["type"] == "haystack.components.preprocessors.semantic_resolver.SemanticResolver"
        assert serialized["init_parameters"]["url"] == "http://custom-tei:8080/similarity"
        assert serialized["init_parameters"]["threshold"] == 0.85
        assert serialized["init_parameters"]["metadata_key"] == "ents"
        assert serialized["init_parameters"]["canonical_name_strategy"] == "longest"
        assert serialized["init_parameters"]["batch_size"] == 16

        restored = SemanticResolver.from_dict(serialized)
        assert restored.url == resolver.url
        assert restored.threshold == resolver.threshold
        assert restored.metadata_key == resolver.metadata_key
        assert restored.canonical_name_strategy == resolver.canonical_name_strategy
        assert restored.batch_size == resolver.batch_size

    def test_warm_up_and_close(self):
        resolver = SemanticResolver()
        assert resolver._client is None
        resolver.warm_up()
        assert isinstance(resolver._client, httpx.Client)
        resolver.close()
        assert resolver._client is None

    @pytest.mark.asyncio
    async def test_warm_up_async_and_close_async(self):
        resolver = SemanticResolver()
        assert resolver._async_client is None
        await resolver.warm_up_async()
        assert isinstance(resolver._async_client, httpx.AsyncClient)
        await resolver.close_async()
        assert resolver._async_client is None

    def test_context_manager(self):
        with SemanticResolver() as resolver:
            assert isinstance(resolver._client, httpx.Client)
        assert resolver._client is None

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        async with SemanticResolver() as resolver:
            assert isinstance(resolver._async_client, httpx.AsyncClient)
        assert resolver._async_client is None

    def test_run_rejects_invalid_documents(self):
        resolver = SemanticResolver()
        with pytest.raises(TypeError, match="SemanticResolver expects a List of Documents as input"):
            resolver.run(documents="not-a-list")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="SemanticResolver expects a List of Documents as input"):
            resolver.run(documents=["not-a-document"])  # type: ignore[list-item]

        # Validates all elements in mixed lists
        with pytest.raises(TypeError, match="SemanticResolver expects a List of Documents as input"):
            resolver.run(documents=[Document(content="valid"), "not-a-document"])  # type: ignore[list-item]

    @pytest.mark.asyncio
    async def test_run_async_rejects_invalid_documents(self):
        resolver = SemanticResolver()
        with pytest.raises(TypeError, match="SemanticResolver expects a List of Documents as input"):
            await resolver.run_async(documents="not-a-list")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="SemanticResolver expects a List of Documents as input"):
            await resolver.run_async(documents=["not-a-document"])  # type: ignore[list-item]

        # Validates all elements in mixed lists
        with pytest.raises(TypeError, match="SemanticResolver expects a List of Documents as input"):
            await resolver.run_async(documents=[Document(content="valid"), "not-a-document"])  # type: ignore[list-item]

    def test_run_empty_documents(self):
        resolver = SemanticResolver()
        result = resolver.run(documents=[])
        assert result == {"documents": []}

    def test_run_documents_without_entities_metadata(self):
        resolver = SemanticResolver()
        docs = [
            Document(id="doc1", content="Hello world", meta={"source": "book"}),
            Document(id="doc2", content="Foo bar", meta={}),
        ]
        result = resolver.run(documents=docs)
        assert len(result["documents"]) == 2
        assert result["documents"][0].id == "doc1"
        assert result["documents"][0].meta == {"source": "book"}
        assert result["documents"][1].id == "doc2"
        assert result["documents"][1].meta == {}

    def test_run_single_entity_no_http_call(self):
        resolver = SemanticResolver()
        docs = [
            Document(
                id="doc1",
                content="Deepset is in Berlin.",
                meta={"entities": [{"entity": "deepset", "entity_type": "company"}]},
            )
        ]
        result = resolver.run(documents=docs)
        assert result["documents"][0].meta["entities"] == [
            {"entity": "deepset", "entity_type": "company", "original_entity": "deepset"}
        ]

    def test_run_merges_similar_entities(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            source = json["inputs"]["source_sentence"]
            sentences = json["inputs"]["sentences"]
            scores = []
            for target in sentences:
                if (source, target) in [("deepset", "deepset GmbH"), ("deepset GmbH", "deepset")]:
                    scores.append(0.95)
                elif "Google" in source or "Google" in target:
                    scores.append(0.20)
                else:
                    scores.append(0.10)
            return httpx.Response(200, json=scores, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(
                id="d1",
                content="deepset created Haystack.",
                meta={"entities": [{"entity": "deepset", "entity_type": "company"}]},
            ),
            Document(
                id="d2",
                content="deepset GmbH was founded in Berlin.",
                meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]},
            ),
            Document(
                id="d3",
                content="Google builds AI models.",
                meta={"entities": [{"entity": "Google", "entity_type": "company"}]},
            ),
        ]

        resolver = SemanticResolver(threshold=0.85)
        result = resolver.run(documents=docs)
        out_docs = result["documents"]

        # deepset and deepset GmbH merged to 'deepset' (tie-breaker shortest)
        assert out_docs[0].meta["entities"][0]["entity"] == "deepset"
        assert out_docs[0].meta["entities"][0]["original_entity"] == "deepset"

        assert out_docs[1].meta["entities"][0]["entity"] == "deepset"
        assert out_docs[1].meta["entities"][0]["original_entity"] == "deepset GmbH"

        # Google remains unchanged
        assert out_docs[2].meta["entities"][0]["entity"] == "Google"
        assert out_docs[2].meta["entities"][0]["original_entity"] == "Google"

    def test_rerun_preserves_original_entity(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(id="d1", meta={"entities": [{"entity": "deepset", "entity_type": "company"}]}),
            Document(id="d2", meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]}),
        ]

        resolver = SemanticResolver(threshold=0.85)
        first_result = resolver.run(docs)["documents"]
        assert first_result[1].meta["entities"][0]["entity"] == "deepset"
        assert first_result[1].meta["entities"][0]["original_entity"] == "deepset GmbH"

        # Repeated run must preserve the original alias instead of overwriting with the resolved name
        second_result = resolver.run(first_result)["documents"]
        assert second_result[1].meta["entities"][0]["entity"] == "deepset"
        assert second_result[1].meta["entities"][0]["original_entity"] == "deepset GmbH"

    def test_run_string_entities(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.92], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(id="d1", content="One", meta={"entities": ["Python", "Python 3"]}),
            Document(id="d2", content="Two", meta={"entities": ["Python 3"]}),
        ]

        resolver = SemanticResolver(threshold=0.85)
        result = resolver.run(documents=docs)
        assert result["documents"][0].meta["entities"] == ["Python 3"]
        assert result["documents"][1].meta["entities"] == ["Python 3"]

    def test_canonical_name_strategies(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95, 0.95], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(id="1", meta={"entities": [{"entity": "deepset", "entity_type": "company"}]}),
            Document(id="2", meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]}),
            Document(id="3", meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]}),
            Document(id="4", meta={"entities": [{"entity": "deepset software", "entity_type": "company"}]}),
        ]

        # most_frequent: "deepset GmbH" (count 2)
        r_freq = SemanticResolver(canonical_name_strategy="most_frequent")
        res_freq = r_freq.run(docs)
        assert res_freq["documents"][0].meta["entities"][0]["entity"] == "deepset GmbH"

        # shortest: "deepset" (length 7)
        r_short = SemanticResolver(canonical_name_strategy="shortest")
        res_short = r_short.run(docs)
        assert res_short["documents"][0].meta["entities"][0]["entity"] == "deepset"

        # longest: "deepset software" (length 16)
        r_long = SemanticResolver(canonical_name_strategy="longest")
        res_long = r_long.run(docs)
        assert res_long["documents"][0].meta["entities"][0]["entity"] == "deepset software"

        # first: "deepset" (appeared in doc 1)
        r_first = SemanticResolver(canonical_name_strategy="first")
        res_first = r_first.run(docs)
        assert res_first["documents"][0].meta["entities"][0]["entity"] == "deepset"

    def test_first_strategy_tracked_per_entity_type(self, monkeypatch):
        # Verify that "first" canonical strategy tracks order per entity type independently.
        # "Alpha" appears in doc 1 under type_B (first in type_B).
        # "Beta" appears in doc 1 under type_A (first in type_A).
        # "Alpha" appears in doc 2 under type_A (second in type_A, similar to Beta).
        # Because Beta appeared first in type_A, Beta must win for type_A despite Alpha appearing earlier globally.
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(
                id="1",
                meta={
                    "entities": [
                        {"entity": "Alpha", "entity_type": "type_B"},
                        {"entity": "Beta", "entity_type": "type_A"},
                    ]
                },
            ),
            Document(id="2", meta={"entities": [{"entity": "Alpha", "entity_type": "type_A"}]}),
        ]

        resolver = SemanticResolver(canonical_name_strategy="first", threshold=0.85)
        res = resolver.run(docs)["documents"]

        # In type_A, "Beta" was seen first (index 0 for type_A), so "Beta" must be canonical
        assert res[0].meta["entities"][1]["entity"] == "Beta"
        assert res[1].meta["entities"][0]["entity"] == "Beta"

    def test_group_by_type_separation(self, monkeypatch):
        call_count = 0

        def mock_post(client, url, json=None, headers=None):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=[1.0], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(meta={"entities": [{"entity": "Apple", "entity_type": "company"}]}),
            Document(meta={"entities": [{"entity": "Apple", "entity_type": "food"}]}),
        ]

        resolver = SemanticResolver(group_by_type=True)
        result = resolver.run(docs)
        # Because each group has only 1 entity, no similarity call is made!
        assert call_count == 0
        assert result["documents"][0].meta["entities"][0]["entity_type"] == "company"
        assert result["documents"][1].meta["entities"][0]["entity_type"] == "food"

    def test_transitive_clustering(self, monkeypatch):
        # A ~ B (0.90) and B ~ C (0.90), but A ~ C (0.10)
        # Verify that transitive clustering connects all 3, but the API compares B against C
        queried_pairs = []

        def mock_post(client, url, json=None, headers=None):
            source = json["inputs"]["source_sentence"]
            sentences = json["inputs"]["sentences"]
            scores = []
            for target in sentences:
                queried_pairs.append((source, target))
                pair = {source, target}
                if pair in [{"A", "B"}, {"B", "C"}]:
                    scores.append(0.90)
                else:
                    scores.append(0.10)
            return httpx.Response(200, json=scores, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(meta={"entities": [{"entity": "A", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "B", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "C", "entity_type": "t"}]}),
        ]

        resolver = SemanticResolver(threshold=0.85)
        result = resolver.run(docs)

        # A is compared to B and C, then B is compared to C
        assert ("A", "B") in queried_pairs
        assert ("A", "C") in queried_pairs
        assert ("B", "C") in queried_pairs

        # All 3 documents resolve to the same canonical entity
        canonical_entities = {d.meta["entities"][0]["entity"] for d in result["documents"]}
        assert len(canonical_entities) == 1

    def test_candidate_reduction(self, monkeypatch):
        # A-B and A-C both exceed the threshold in the first batch;
        # expect only one source API call, from A.
        queried_sources = []

        def mock_post(client, url, json=None, headers=None):
            source = json["inputs"]["source_sentence"]
            sentences = json["inputs"]["sentences"]
            queried_sources.append(source)
            return httpx.Response(200, json=[0.95] * len(sentences), request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(meta={"entities": [{"entity": "A", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "B", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "C", "entity_type": "t"}]}),
        ]

        resolver = SemanticResolver(threshold=0.85)
        resolver.run(docs)

        # Since A was compared with both B and C and united them into one cluster,
        # source B never needs to make an API call for C.
        assert queried_sources == ["A"]

    def test_input_immutability(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        orig_entity_1 = {"entity": "deepset", "entity_type": "company"}
        orig_entity_2 = {"entity": "deepset GmbH", "entity_type": "company"}
        docs = [
            Document(id="d1", meta={"entities": [orig_entity_1]}),
            Document(id="d2", meta={"entities": [orig_entity_2]}),
        ]

        resolver = SemanticResolver()
        resolver.run(docs)

        # Original input dicts must not be changed
        assert orig_entity_1 == {"entity": "deepset", "entity_type": "company"}
        assert orig_entity_2 == {"entity": "deepset GmbH", "entity_type": "company"}
        assert "original_entity" not in orig_entity_1
        assert "original_entity" not in orig_entity_2

    def test_deduplicate_different_aliases_in_same_document(self, monkeypatch):
        # When two different aliases in the same document resolve to the same canonical entity,
        # they must be deduplicated into a single entity entry.
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(
                meta={
                    "entities": [
                        {"entity": "deepset", "entity_type": "company"},
                        {"entity": "deepset", "entity_type": "company"},
                        {"entity": "deepset GmbH", "entity_type": "company"},
                    ]
                }
            )
        ]

        resolver = SemanticResolver(deduplicate_entities=True, threshold=0.85)
        res = resolver.run(docs)
        entities = res["documents"][0].meta["entities"]
        assert len(entities) == 1
        assert entities[0]["entity"] == "deepset"
        assert entities[0]["original_entity"] == "deepset"

    def test_tei_error_handling(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(500, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(meta={"entities": [{"entity": "E1", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "E2", "entity_type": "t"}]}),
        ]

        resolver = SemanticResolver()
        with pytest.raises(httpx.HTTPStatusError):
            resolver.run(docs)

    def test_tei_malformed_response(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json={"error": "not a list"}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        docs = [
            Document(meta={"entities": [{"entity": "E1", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "E2", "entity_type": "t"}]}),
        ]

        resolver = SemanticResolver()
        with pytest.raises(ValueError, match="Unexpected response format from TEI similarity endpoint"):
            resolver.run(docs)

    @pytest.mark.asyncio
    async def test_run_async(self, monkeypatch):
        async def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        docs = [
            Document(meta={"entities": [{"entity": "deepset", "entity_type": "company"}]}),
            Document(meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]}),
        ]

        resolver = SemanticResolver()
        res = await resolver.run_async(docs)
        assert res["documents"][0].meta["entities"][0]["entity"] == "deepset"
        assert res["documents"][1].meta["entities"][0]["entity"] == "deepset"

    def test_pipeline_integration(self, monkeypatch):
        pipeline = Pipeline()
        pipeline.add_component("resolver", SemanticResolver(threshold=0.88))
        serialized = pipeline.to_dict()
        assert "resolver" in serialized["components"]
        assert serialized["components"]["resolver"]["init_parameters"]["threshold"] == 0.88

    def test_init_rejects_nan_threshold_and_timeout(self):
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            SemanticResolver(threshold=float("nan"))
        with pytest.raises(ValueError, match="timeout must be greater than 0"):
            SemanticResolver(timeout=float("nan"))

    def test_init_rejects_empty_type_key(self):
        with pytest.raises(ValueError, match="type_key cannot be empty"):
            SemanticResolver(type_key="")

    def test_http_client_kwargs_timeout_override(self):
        # http_client_kwargs with timeout should not raise TypeError when warming up
        resolver = SemanticResolver(timeout=30.0, http_client_kwargs={"timeout": 45.0})
        resolver.warm_up()
        assert resolver._client is not None
        assert resolver._client.timeout.read == 45.0
        resolver.close()

    def test_tei_rejects_invalid_similarity_scores(self, monkeypatch):
        # Rejects boolean, string, NaN, Inf in TEI similarity response
        docs = [
            Document(meta={"entities": [{"entity": "E1", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "E2", "entity_type": "t"}]}),
        ]
        resolver = SemanticResolver()

        # Boolean
        monkeypatch.setattr(
            httpx.Client,
            "post",
            lambda *a, **kw: httpx.Response(200, json=[True], request=httpx.Request("POST", "http://test")),
        )
        with pytest.raises(ValueError, match="Invalid similarity score returned by TEI similarity endpoint"):
            resolver.run(docs)

        # Non-numeric string
        monkeypatch.setattr(
            httpx.Client,
            "post",
            lambda *a, **kw: httpx.Response(200, json=["invalid"], request=httpx.Request("POST", "http://test")),
        )
        with pytest.raises(ValueError, match="Invalid similarity score returned by TEI similarity endpoint"):
            resolver.run(docs)

        # NaN
        monkeypatch.setattr(
            httpx.Client,
            "post",
            lambda *a, **kw: httpx.Response(200, content=b"[NaN]", request=httpx.Request("POST", "http://test")),
        )
        with pytest.raises(ValueError, match="Non-finite similarity score returned by TEI similarity endpoint"):
            resolver.run(docs)

        # Inf
        monkeypatch.setattr(
            httpx.Client,
            "post",
            lambda *a, **kw: httpx.Response(200, content=b"[Infinity]", request=httpx.Request("POST", "http://test")),
        )
        with pytest.raises(ValueError, match="Non-finite similarity score returned by TEI similarity endpoint"):
            resolver.run(docs)

    def test_unhashable_entity_types_handled(self, monkeypatch):
        monkeypatch.setattr(
            httpx.Client,
            "post",
            lambda *a, **kw: httpx.Response(200, json=[0.95], request=httpx.Request("POST", "http://test")),
        )
        docs = [
            Document(meta={"entities": [{"entity": "Apple", "entity_type": ["company"]}]}),
            Document(meta={"entities": [{"entity": "Apple Inc", "entity_type": ["company"]}]}),
        ]
        resolver = SemanticResolver(threshold=0.85)
        res = resolver.run(docs)
        assert res["documents"][0].meta["entities"][0]["entity"] == "Apple"
        assert res["documents"][1].meta["entities"][0]["entity"] == "Apple"

        # Also handles nested unhashable structures recursively
        docs_nested = [
            Document(meta={"entities": [{"entity": "Google", "entity_type": [["tech", "search"]]}]}),
            Document(meta={"entities": [{"entity": "Google LLC", "entity_type": [["tech", "search"]]}]}),
        ]
        res_nested = resolver.run(docs_nested)
        assert res_nested["documents"][0].meta["entities"][0]["entity"] == "Google"
        assert res_nested["documents"][1].meta["entities"][0]["entity"] == "Google"

    def test_batch_size_chunking_with_many_targets(self, monkeypatch):
        chunk_sizes = []

        def mock_post(client, url, json=None, headers=None):
            chunk = json["inputs"]["sentences"]
            chunk_sizes.append(len(chunk))
            return httpx.Response(200, json=[0.1] * len(chunk), request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)
        docs = [Document(meta={"entities": [{"entity": f"Entity_{i}", "entity_type": "t"}]}) for i in range(5)]
        resolver = SemanticResolver(batch_size=2)
        resolver.run(docs)
        # Source 0 has 4 targets, chunked with batch_size=2 -> calls of size 2
        assert 2 in chunk_sizes
        assert len(chunk_sizes) > 1

    def test_deduplicate_entities_preserves_all_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            httpx.Client,
            "post",
            lambda *a, **kw: httpx.Response(200, json=[0.95], request=httpx.Request("POST", "http://test")),
        )
        docs = [
            Document(
                meta={
                    "entities": [
                        {"entity": "deepset", "entity_type": "company", "confidence": 0.95, "source": "src1"},
                        {"entity": "deepset GmbH", "entity_type": "company", "confidence": 0.88, "source": "src2"},
                    ]
                }
            )
        ]
        resolver = SemanticResolver(deduplicate_entities=False, threshold=0.85)
        res = resolver.run(docs)
        entities = res["documents"][0].meta["entities"]
        assert len(entities) == 2
        assert entities[0]["entity"] == "deepset"
        assert entities[0]["confidence"] == 0.95
        assert entities[0]["source"] == "src1"
        assert entities[1]["entity"] == "deepset"
        assert entities[1]["confidence"] == 0.88
        assert entities[1]["source"] == "src2"

    @pytest.mark.asyncio
    async def test_async_tei_error_handling(self, monkeypatch):
        async def mock_post(*a, **kw):
            return httpx.Response(500, request=httpx.Request("POST", "http://test"))

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
        docs = [
            Document(meta={"entities": [{"entity": "E1", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "E2", "entity_type": "t"}]}),
        ]
        resolver = SemanticResolver()
        with pytest.raises(httpx.HTTPStatusError):
            await resolver.run_async(docs)

    @pytest.mark.asyncio
    async def test_async_malformed_response(self, monkeypatch):
        async def mock_post(*a, **kw):
            return httpx.Response(200, json={"error": "bad"}, request=httpx.Request("POST", "http://test"))

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
        docs = [
            Document(meta={"entities": [{"entity": "E1", "entity_type": "t"}]}),
            Document(meta={"entities": [{"entity": "E2", "entity_type": "t"}]}),
        ]
        resolver = SemanticResolver()
        with pytest.raises(ValueError, match="Unexpected response format from TEI similarity endpoint"):
            await resolver.run_async(docs)

    @pytest.mark.asyncio
    async def test_sync_and_async_parity(self, monkeypatch):
        def mock_post(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95], request=httpx.Request("POST", url))

        async def mock_post_async(client, url, json=None, headers=None):
            return httpx.Response(200, json=[0.95], request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)
        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_async)

        docs_sync = [
            Document(meta={"entities": [{"entity": "deepset", "entity_type": "company"}]}),
            Document(meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]}),
        ]
        docs_async = [
            Document(meta={"entities": [{"entity": "deepset", "entity_type": "company"}]}),
            Document(meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]}),
        ]

        resolver = SemanticResolver(threshold=0.85)
        res_sync = resolver.run(docs_sync)
        res_async = await resolver.run_async(docs_async)

        assert res_sync["documents"][0].meta == res_async["documents"][0].meta
        assert res_sync["documents"][1].meta == res_async["documents"][1].meta
