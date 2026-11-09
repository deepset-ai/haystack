# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from copy import copy, deepcopy
from typing import Any, Literal

import httpx

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


class _UnionFind:
    """Disjoint-set / Union-Find data structure for clustering entities."""

    def __init__(self, elements: list[str]) -> None:
        self.parent = {elem: elem for elem in elements}
        self.rank = dict.fromkeys(elements, 0)

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


@component
class SemanticResolver:
    """
    Resolves semantically equivalent entities across document metadata using Text Embeddings Inference (TEI).

    When extracting entities from documents (for example, using `LLMMetadataExtractor`), the same real-world entity
    may appear under slight variations (e.g., 'deepset' and 'deepset GmbH'). If ingested into a knowledge graph or
    retrieval pipeline, these variations might lead to duplicate or fragmented nodes.

    `SemanticResolver` inspects entity lists stored in document metadata and queries a Hugging Face Text Embeddings
    Inference (TEI) `/similarity` endpoint to determine pairwise semantic similarity. Entities with similarity
    at or above a configurable threshold are clustered and normalized under a single canonical name.

    ### Usage example:

    ```python
    from haystack import Document
    from haystack.components.preprocessors import SemanticResolver

    docs = [
        Document(
            content="deepset created Haystack.",
            meta={"entities": [{"entity": "deepset", "entity_type": "company"}]},
        ),
        Document(
            content="deepset GmbH is based in Berlin.",
            meta={"entities": [{"entity": "deepset GmbH", "entity_type": "company"}]},
        ),
    ]

    resolver = SemanticResolver(url="http://localhost:8080", threshold=0.85)
    # When TEI server is running:
    # result = resolver.run(documents=docs)
    # result["documents"][0].meta["entities"][0]["entity"] == "deepset"
    # result["documents"][1].meta["entities"][0]["entity"] == "deepset"
    ```
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Secret | None = None,
        threshold: float = 0.8,
        metadata_key: str = "entities",
        entity_key: str = "entity",
        type_key: str = "entity_type",
        group_by_type: bool = True,
        canonical_name_strategy: Literal["most_frequent", "shortest", "longest", "first"] = "most_frequent",
        keep_original_entity_name: bool = True,
        deduplicate_entities: bool = True,
        batch_size: int = 64,
        timeout: float = 30.0,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the SemanticResolver component.

        :param url: The base URL of the Text Embeddings Inference (TEI) service. If `/similarity` is omitted,
            it is automatically appended.
        :param api_key: The Hugging Face token or API key for authentication. If not provided, it defaults to the
            `HF_TOKEN` environment variable.
        :param threshold: The similarity threshold (between 0.0 and 1.0) above which two entities are clustered.
        :param metadata_key: The metadata dictionary key containing entity information. Defaults to `"entities"`.
        :param entity_key: The dictionary key for the entity name inside entity objects. Defaults to `"entity"`.
        :param type_key: The dictionary key for the entity type inside entity objects. Defaults to `"entity_type"`.
        :param group_by_type: If `True`, only compares entities having identical entity types. Defaults to `True`.
        :param canonical_name_strategy: Strategy to pick the canonical representative of a merged cluster.
            Options are:
            - `"most_frequent"`: The name occurring most frequently across input documents.
            - `"shortest"`: The shortest name by character count.
            - `"longest"`: The longest name by character count.
            - `"first"`: The first name encountered in the input documents.
            Defaults to `"most_frequent"`.
        :param keep_original_entity_name: If `True`, preserves the pre-resolution entity name in
            `original_<entity_key>`. Defaults to `True`.
        :param deduplicate_entities: If `True`, removes duplicate entities within each document's metadata list
            after resolution. Defaults to `True`.
        :param batch_size: Maximum number of target sentences to send in a single TEI `/similarity` request.
            Defaults to 64.
        :param timeout: HTTP request timeout in seconds. Defaults to 30.0.
        :param http_client_kwargs: Optional keyword arguments passed to the underlying `httpx.Client`.
        :raises ValueError: If parameters are invalid (e.g. threshold out of bounds).
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")
        if timeout <= 0:
            raise ValueError(f"timeout must be greater than 0, got {timeout}")
        valid_strategies = ("most_frequent", "shortest", "longest", "first")
        if canonical_name_strategy not in valid_strategies:
            raise ValueError(
                f"canonical_name_strategy must be one of {valid_strategies}, got '{canonical_name_strategy}'"
            )
        if not metadata_key:
            raise ValueError("metadata_key cannot be empty")
        if not entity_key:
            raise ValueError("entity_key cannot be empty")

        normalized_url = url.rstrip("/")
        if not normalized_url.endswith("/similarity"):
            normalized_url = f"{normalized_url}/similarity"

        self.url = normalized_url
        self.api_key = api_key or Secret.from_env_var("HF_TOKEN", strict=False)
        self.threshold = threshold
        self.metadata_key = metadata_key
        self.entity_key = entity_key
        self.type_key = type_key
        self.group_by_type = group_by_type
        self.canonical_name_strategy = canonical_name_strategy
        self.keep_original_entity_name = keep_original_entity_name
        self.deduplicate_entities = deduplicate_entities
        self.batch_size = batch_size
        self.timeout = timeout
        self.http_client_kwargs = http_client_kwargs or {}

        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def warm_up(self) -> None:
        """Initializes the synchronous HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, **self.http_client_kwargs)

    async def warm_up_async(self) -> None:  # noqa: RUF029
        """Initializes the asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout, **self.http_client_kwargs)

    def close(self) -> None:
        """Closes the synchronous HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Closes the asynchronous HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> "SemanticResolver":
        self.warm_up()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "SemanticResolver":
        await self.warm_up_async()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            url=self.url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            threshold=self.threshold,
            metadata_key=self.metadata_key,
            entity_key=self.entity_key,
            type_key=self.type_key,
            group_by_type=self.group_by_type,
            canonical_name_strategy=self.canonical_name_strategy,
            keep_original_entity_name=self.keep_original_entity_name,
            deduplicate_entities=self.deduplicate_entities,
            batch_size=self.batch_size,
            timeout=self.timeout,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticResolver":
        """
        Deserializes the component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key is not None:
            token = self.api_key.resolve_value()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _extract_entities_from_documents(
        self, documents: list[Document]
    ) -> tuple[dict[Any, dict[str, int]], dict[str, int]]:
        """
        Extracts entity occurrences and order of appearance from documents.

        :param documents: The list of documents.
        :returns: A tuple of:
            - mapping: `entity_type -> {entity_name: count}`
            - first_seen: `{entity_name: first_seen_index}`
        """
        occurrences: dict[Any, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        first_seen: dict[str, int] = {}
        seen_counter = 0

        for doc in documents:
            if not doc.meta or self.metadata_key not in doc.meta:
                continue
            entities = doc.meta[self.metadata_key]
            if not isinstance(entities, list):
                continue

            for item in entities:
                if isinstance(item, dict):
                    name = item.get(self.entity_key)
                    etype = item.get(self.type_key) if self.group_by_type else None
                elif isinstance(item, str):
                    name = item
                    etype = None
                else:
                    continue

                if name is not None and isinstance(name, str):
                    occurrences[etype][name] += 1
                    if name not in first_seen:
                        first_seen[name] = seen_counter
                        seen_counter += 1

        return occurrences, first_seen

    def _select_canonical_name(self, cluster: list[str], counts: dict[str, int], first_seen: dict[str, int]) -> str:
        """
        Selects a deterministic canonical name for a cluster of equivalent entity names.

        :param cluster: List of entity names in the cluster.
        :param counts: Frequency count of each entity name.
        :param first_seen: Index of first appearance for each entity name.
        :returns: The canonical entity name.
        """
        if len(cluster) == 1:
            return cluster[0]

        if self.canonical_name_strategy == "most_frequent":
            # Higher count first, then shorter string, then alphabetical
            return min(cluster, key=lambda name: (-counts.get(name, 0), len(name), name))
        elif self.canonical_name_strategy == "shortest":
            # Shorter string first, then higher count, then alphabetical
            return min(cluster, key=lambda name: (len(name), -counts.get(name, 0), name))
        elif self.canonical_name_strategy == "longest":
            # Longer string first, then higher count, then alphabetical
            return min(cluster, key=lambda name: (-len(name), -counts.get(name, 0), name))
        elif self.canonical_name_strategy == "first":
            # First encountered in documents, then alphabetical
            return min(cluster, key=lambda name: (first_seen.get(name, float("inf")), name))

        return cluster[0]

    def _resolve_clusters(
        self, names: list[str], uf: _UnionFind, counts: dict[str, int], first_seen: dict[str, int]
    ) -> dict[str, str]:
        """
        Groups names by connected components and assigns each name to its canonical representative.

        :param names: List of unique entity names in the group.
        :param uf: Union-Find instance with clustered components.
        :param counts: Frequency counts.
        :param first_seen: First-seen indices.
        :returns: Mapping from original name to canonical name.
        """
        clusters: dict[str, list[str]] = defaultdict(list)
        for name in names:
            clusters[uf.find(name)].append(name)

        name_to_canonical: dict[str, str] = {}
        for cluster in clusters.values():
            canonical = self._select_canonical_name(cluster, counts, first_seen)
            for name in cluster:
                name_to_canonical[name] = canonical

        return name_to_canonical

    def _apply_resolution_to_documents(
        self, documents: list[Document], resolution_map: dict[tuple[str, Any], str]
    ) -> list[Document]:
        """
        Creates new Document instances with normalized entity metadata.

        :param documents: The input documents.
        :param resolution_map: Mapping `(original_name, entity_type) -> canonical_name`.
        :returns: List of new Document instances with resolved entities.
        """
        resolved_docs: list[Document] = []

        for doc in documents:
            if not doc.meta or self.metadata_key not in doc.meta:
                resolved_docs.append(
                    Document(
                        id=doc.id,
                        content=doc.content,
                        blob=doc.blob,
                        meta=deepcopy(doc.meta),
                        score=doc.score,
                        embedding=doc.embedding,
                        sparse_embedding=doc.sparse_embedding,
                    )
                )
                continue

            new_meta = deepcopy(doc.meta)
            raw_entities = new_meta.get(self.metadata_key)
            if not isinstance(raw_entities, list):
                resolved_docs.append(
                    Document(
                        id=doc.id,
                        content=doc.content,
                        blob=doc.blob,
                        meta=new_meta,
                        score=doc.score,
                        embedding=doc.embedding,
                        sparse_embedding=doc.sparse_embedding,
                    )
                )
                continue

            updated_entities: list[Any] = []
            for item in raw_entities:
                if isinstance(item, dict):
                    name = item.get(self.entity_key)
                    etype = item.get(self.type_key) if self.group_by_type else None
                    if name is not None and isinstance(name, str):
                        canonical = resolution_map.get((name, etype), name)
                        new_item = copy(item)
                        new_item[self.entity_key] = canonical
                        if self.keep_original_entity_name:
                            new_item[f"original_{self.entity_key}"] = name
                        updated_entities.append(new_item)
                    else:
                        updated_entities.append(copy(item))
                elif isinstance(item, str):
                    canonical = resolution_map.get((item, None), item)
                    updated_entities.append(canonical)
                else:
                    updated_entities.append(copy(item))

            if self.deduplicate_entities:
                seen_entities: list[Any] = []
                for entity in updated_entities:
                    if entity not in seen_entities:
                        seen_entities.append(entity)
                updated_entities = seen_entities

            new_meta[self.metadata_key] = updated_entities

            resolved_docs.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    blob=doc.blob,
                    meta=new_meta,
                    score=doc.score,
                    embedding=doc.embedding,
                    sparse_embedding=doc.sparse_embedding,
                )
            )

        return resolved_docs

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Resolves similar entities across the provided documents.

        :param documents: List of Documents containing extracted entities in metadata.
        :returns: A dictionary with the following key:
            - `documents`: List of Documents with normalized entity names.
        :raises TypeError: If `documents` is not a list of Documents.
        :raises ValueError: If the TEI response format is invalid.
        :raises httpx.HTTPError: If the request to the TEI service fails.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("SemanticResolver expects a List of Documents as input.")

        if not documents:
            return {"documents": []}

        self.warm_up()
        assert self._client is not None

        occurrences, first_seen = self._extract_entities_from_documents(documents)
        resolution_map: dict[tuple[str, Any], str] = {}
        headers = self._get_headers()

        for etype, name_counts in occurrences.items():
            names = list(name_counts.keys())
            if len(names) <= 1:
                if len(names) == 1:
                    resolution_map[(names[0], etype)] = names[0]
                continue

            uf = _UnionFind(names)
            for i in range(len(names)):
                source = names[i]
                candidate_targets = [target for target in names[i + 1 :] if uf.find(source) != uf.find(target)]
                if not candidate_targets:
                    continue

                for chunk_start in range(0, len(candidate_targets), self.batch_size):
                    chunk = candidate_targets[chunk_start : chunk_start + self.batch_size]
                    payload = {"inputs": {"source_sentence": source, "sentences": chunk}}
                    response = self._client.post(self.url, json=payload, headers=headers)
                    response.raise_for_status()
                    scores = response.json()

                    if not isinstance(scores, list) or len(scores) != len(chunk):
                        raise ValueError(
                            f"Unexpected response format from TEI similarity endpoint: expected list of "
                            f"{len(chunk)} float scores, got: {scores}"
                        )

                    for target, score in zip(chunk, scores, strict=True):
                        if float(score) >= self.threshold:
                            uf.union(source, target)

            cluster_map = self._resolve_clusters(names, uf, name_counts, first_seen)
            for name, canonical in cluster_map.items():
                resolution_map[(name, etype)] = canonical

        resolved_docs = self._apply_resolution_to_documents(documents, resolution_map)
        return {"documents": resolved_docs}

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously resolves similar entities across the provided documents.

        :param documents: List of Documents containing extracted entities in metadata.
        :returns: A dictionary with the following key:
            - `documents`: List of Documents with normalized entity names.
        :raises TypeError: If `documents` is not a list of Documents.
        :raises ValueError: If the TEI response format is invalid.
        :raises httpx.HTTPError: If the request to the TEI service fails.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("SemanticResolver expects a List of Documents as input.")

        if not documents:
            return {"documents": []}

        await self.warm_up_async()
        assert self._async_client is not None

        occurrences, first_seen = self._extract_entities_from_documents(documents)
        resolution_map: dict[tuple[str, Any], str] = {}
        headers = self._get_headers()

        for etype, name_counts in occurrences.items():
            names = list(name_counts.keys())
            if len(names) <= 1:
                if len(names) == 1:
                    resolution_map[(names[0], etype)] = names[0]
                continue

            uf = _UnionFind(names)
            for i in range(len(names)):
                source = names[i]
                candidate_targets = [target for target in names[i + 1 :] if uf.find(source) != uf.find(target)]
                if not candidate_targets:
                    continue

                for chunk_start in range(0, len(candidate_targets), self.batch_size):
                    chunk = candidate_targets[chunk_start : chunk_start + self.batch_size]
                    payload = {"inputs": {"source_sentence": source, "sentences": chunk}}
                    response = await self._async_client.post(self.url, json=payload, headers=headers)
                    response.raise_for_status()
                    scores = response.json()

                    if not isinstance(scores, list) or len(scores) != len(chunk):
                        raise ValueError(
                            f"Unexpected response format from TEI similarity endpoint: expected list of "
                            f"{len(chunk)} float scores, got: {scores}"
                        )

                    for target, score in zip(chunk, scores, strict=True):
                        if float(score) >= self.threshold:
                            uf.union(source, target)

            cluster_map = self._resolve_clusters(names, uf, name_counts, first_seen)
            for name, canonical in cluster_map.items():
                resolution_map[(name, etype)] = canonical

        resolved_docs = self._apply_resolution_to_documents(documents, resolution_map)
        return {"documents": resolved_docs}
