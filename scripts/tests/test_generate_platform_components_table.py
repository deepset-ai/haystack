# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/generate_platform_components_table.py."""

import ast
import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_platform_components_table import (
    _parse_frontmatter,
    build_mdx,
    compute_module_path,
    has_component_decorator,
    infer_type,
    load_platform_components,
    partner_label_for,
    scan_docs_links,
    scan_for_components,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_class(source: str) -> ast.ClassDef:
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node
    raise AssertionError("No ClassDef found in snippet")


def _make_schema(components: dict) -> dict:
    """Wrap a flat {fqn: {title, family}} dict into the full schema envelope."""
    defs: dict = {}
    for fqn, meta in components.items():
        defs[fqn] = {
            "type": "object",
            "title": meta["title"],
            "properties": {
                "type": {
                    "type": "string",
                    "const": fqn,
                    "origin": fqn,
                    "family": meta.get("family", ""),
                }
            },
        }
    return {"type": "object", "definitions": {"Components": defs}}


# ── has_component_decorator ───────────────────────────────────────────────────


class TestHasComponentDecorator:
    def test_bare_decorator(self):
        node = _parse_class(
            """
            @component
            class MyComponent:
                pass
            """
        )
        assert has_component_decorator(node) is True

    def test_decorator_with_arguments(self):
        node = _parse_class(
            """
            @component(some_flag=True)
            class MyComponent:
                pass
            """
        )
        assert has_component_decorator(node) is True

    def test_decorator_with_positional_argument(self):
        node = _parse_class(
            """
            @component("value")
            class MyComponent:
                pass
            """
        )
        assert has_component_decorator(node) is True

    def test_no_decorator(self):
        node = _parse_class(
            """
            class Plain:
                pass
            """
        )
        assert has_component_decorator(node) is False

    def test_different_decorator(self):
        node = _parse_class(
            """
            @dataclass
            class NotAComponent:
                pass
            """
        )
        assert has_component_decorator(node) is False

    def test_similar_name_not_matched(self):
        """@component_something must NOT be treated as @component."""
        node = _parse_class(
            """
            @component_something
            class AlmostComponent:
                pass
            """
        )
        assert has_component_decorator(node) is False

    def test_similar_name_called_not_matched(self):
        node = _parse_class(
            """
            @component_factory()
            class AlmostComponent:
                pass
            """
        )
        assert has_component_decorator(node) is False

    def test_multiple_decorators_one_matches(self):
        node = _parse_class(
            """
            @some_other_decorator
            @component
            class MyComponent:
                pass
            """
        )
        assert has_component_decorator(node) is True

    def test_inline_word_in_body_not_matched(self):
        node = _parse_class(
            """
            @dataclass
            class NotAComponent:
                component = "value"
            """
        )
        assert has_component_decorator(node) is False

    def test_super_component_bare_matched(self):
        node = _parse_class(
            """
            @super_component
            class MyHybridRetriever:
                pass
            """
        )
        assert has_component_decorator(node) is True

    def test_super_component_called_matched(self):
        node = _parse_class(
            """
            @super_component(some_flag=True)
            class MyHybridRetriever:
                pass
            """
        )
        assert has_component_decorator(node) is True

    def test_super_component_similar_name_not_matched(self):
        node = _parse_class(
            """
            @super_component_factory
            class NotMatched:
                pass
            """
        )
        assert has_component_decorator(node) is False


# ── compute_module_path ───────────────────────────────────────────────────────


class TestComputeModulePath:
    def test_haystack_package_file(self, tmp_path):
        file_path = tmp_path / "haystack" / "components" / "converters" / "csv.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()
        assert compute_module_path(file_path, tmp_path) == "haystack.components.converters.csv"

    def test_haystack_integrations_nested_file(self, tmp_path):
        file_path = (
            tmp_path
            / "integrations"
            / "opensearch"
            / "src"
            / "haystack_integrations"
            / "components"
            / "retrievers"
            / "opensearch"
            / "bm25_retriever.py"
        )
        file_path.parent.mkdir(parents=True)
        file_path.touch()
        result = compute_module_path(file_path, tmp_path)
        assert result == "haystack_integrations.components.retrievers.opensearch.bm25_retriever"

    def test_init_file_strips_dunder_init(self, tmp_path):
        file_path = tmp_path / "haystack" / "components" / "__init__.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()
        assert compute_module_path(file_path, tmp_path) == "haystack.components"

    def test_unrecognised_package_returns_none(self, tmp_path):
        file_path = tmp_path / "tests" / "test_helper.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()
        assert compute_module_path(file_path, tmp_path) is None

    def test_file_outside_repo_root_returns_none(self, tmp_path):
        other = tmp_path / "other" / "haystack" / "something.py"
        other.parent.mkdir(parents=True)
        other.touch()
        assert compute_module_path(other, tmp_path / "nonexistent") is None


# ── scan_for_components ───────────────────────────────────────────────────────


class TestScanForComponents:
    def test_detects_decorated_class(self, tmp_path):
        pkg = tmp_path / "haystack" / "components" / "generators"
        pkg.mkdir(parents=True)
        (pkg / "openai.py").write_text(
            textwrap.dedent(
                """
                @component
                class OpenAIGenerator:
                    pass
                """
            )
        )
        result = scan_for_components(tmp_path / "haystack", tmp_path)
        assert "haystack.components.generators.openai.OpenAIGenerator" in result

    def test_ignores_undecorated_class(self, tmp_path):
        pkg = tmp_path / "haystack" / "utils"
        pkg.mkdir(parents=True)
        (pkg / "helper.py").write_text("class NotAComponent:\n    pass\n")
        assert scan_for_components(tmp_path / "haystack", tmp_path) == set()

    def test_multiple_classes_one_file(self, tmp_path):
        pkg = tmp_path / "haystack" / "components" / "embedders"
        pkg.mkdir(parents=True)
        (pkg / "sentence_transformers.py").write_text(
            textwrap.dedent(
                """
                @component
                class SentenceTransformersDocumentEmbedder:
                    pass

                @component
                class SentenceTransformersTextEmbedder:
                    pass

                class HelperClass:
                    pass
                """
            )
        )
        result = scan_for_components(tmp_path / "haystack", tmp_path)
        assert len(result) == 2


# ── load_platform_components ──────────────────────────────────────────────────


class TestLoadPlatformComponents:
    def test_loads_haystack_components(self, tmp_path):
        schema = _make_schema(
            {
                "haystack.components.generators.openai.OpenAIGenerator": {
                    "title": "OpenAIGenerator",
                    "family": "generators",
                }
            }
        )
        f = tmp_path / "schema.json"
        f.write_text(json.dumps(schema))
        result = load_platform_components(f)
        assert "haystack.components.generators.openai.OpenAIGenerator" in result
        assert result["haystack.components.generators.openai.OpenAIGenerator"]["title"] == "OpenAIGenerator"
        assert result["haystack.components.generators.openai.OpenAIGenerator"]["family"] == "generators"

    def test_loads_integrations_components(self, tmp_path):
        schema = _make_schema(
            {
                "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever": {
                    "title": "OpenSearchBM25Retriever",
                    "family": "retrievers",
                }
            }
        )
        f = tmp_path / "schema.json"
        f.write_text(json.dumps(schema))
        result = load_platform_components(f)
        assert "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever" in result

    def test_excludes_deepset_cloud_custom_nodes(self, tmp_path):
        schema = _make_schema(
            {
                "deepset_cloud_custom_nodes.rankers.nvidia.DeepsetNvidiaRanker": {
                    "title": "DeepsetNvidiaRanker",
                    "family": "rankers",
                }
            }
        )
        f = tmp_path / "schema.json"
        f.write_text(json.dumps(schema))
        assert load_platform_components(f) == {}

    def test_excludes_studio_internal(self, tmp_path):
        schema = _make_schema(
            {"studio_internal.foo.Bar": {"title": "Bar", "family": "generators"}}
        )
        f = tmp_path / "schema.json"
        f.write_text(json.dumps(schema))
        assert load_platform_components(f) == {}

    def test_excludes_haystack_experimental(self, tmp_path):
        schema = _make_schema(
            {
                "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator": {
                    "title": "OpenAIChatGenerator",
                    "family": "generators",
                }
            }
        )
        f = tmp_path / "schema.json"
        f.write_text(json.dumps(schema))
        assert load_platform_components(f) == {}

    def test_excludes_deepl_haystack(self, tmp_path):
        schema = _make_schema(
            {"deepl_haystack.components.DeepLTextTranslator": {"title": "DeepLTextTranslator", "family": "translators"}}
        )
        f = tmp_path / "schema.json"
        f.write_text(json.dumps(schema))
        assert load_platform_components(f) == {}


# ── partner_label_for ─────────────────────────────────────────────────────────


class TestPartnerLabelFor:
    def test_known_partner(self):
        assert (
            partner_label_for(
                "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever"
            )
            == "OpenSearch"
        )

    def test_voyage_embedders_merged(self):
        assert (
            partner_label_for(
                "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder.VoyageTextEmbedder"
            )
            == "Voyage AI"
        )

    def test_voyage_ranker_merged(self):
        assert (
            partner_label_for("haystack_integrations.components.rankers.voyage.ranker.VoyageRanker")
            == "Voyage AI"
        )

    def test_docling_serve_merged_with_docling(self):
        assert (
            partner_label_for(
                "haystack_integrations.components.converters.docling_serve.converter.DoclingServeConverter"
            )
            == "Docling"
        )

    def test_amazon_bedrock_label(self):
        assert (
            partner_label_for(
                "haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator.AmazonBedrockChatGenerator"
            )
            == "Amazon Bedrock"
        )

    def test_unknown_partner_title_cases(self):
        result = partner_label_for("haystack_integrations.components.retrievers.some_new_partner.foo.Bar")
        assert result == "Some New Partner"

    def test_returns_none_for_short_fqn(self):
        assert partner_label_for("haystack_integrations.components") is None


# ── infer_type ────────────────────────────────────────────────────────────────


class TestInferType:
    @pytest.mark.parametrize(
        "family,fqn,expected",
        [
            ("generators", "haystack.components.generators.openai.OpenAIGenerator", "Generator"),
            ("retrievers", "haystack.components.retrievers.memory.InMemoryBM25Retriever", "Retriever"),
            ("embedders", "haystack.components.embedders.foo.Bar", "Embedder"),
            ("converters", "haystack.components.converters.csv.CSVToDocument", "Converter"),
            ("document_stores", "haystack.document_stores.foo.Bar", "Document Store"),
            # Falls back to FQN path inspection when family is missing
            (None, "haystack.components.rankers.foo.Bar", "Ranker"),
            ("", "haystack.components.writers.foo.Bar", "Writer"),
            # Unknown returns Component
            (None, "haystack.components.misc.foo.Bar", "Component"),
        ],
    )
    def test_infer_type(self, family, fqn, expected):
        assert infer_type(family, fqn) == expected


# ── Cross-reference logic in build_mdx ───────────────────────────────────────


_PLATFORM = {
    "haystack.components.generators.openai.OpenAIGenerator": {
        "title": "OpenAIGenerator",
        "family": "generators",
    },
    "haystack.components.retrievers.memory.InMemoryBM25Retriever": {
        "title": "InMemoryBM25Retriever",
        "family": "retrievers",
    },
    "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever": {
        "title": "OpenSearchBM25Retriever",
        "family": "retrievers",
    },
}

_SOURCE = {
    "haystack.components.generators.openai.OpenAIGenerator",
    "haystack.components.retrievers.memory.InMemoryBM25Retriever",
    "haystack_integrations.components.retrievers.opensearch.bm25_retriever.OpenSearchBM25Retriever",
}


class TestBuildMdxCrossReference:
    def test_schema_component_in_source_appears(self):
        mdx = build_mdx(_PLATFORM, _SOURCE)
        assert "InMemoryBM25Retriever" in mdx

    def test_component_not_in_source_excluded(self):
        # CSVToDocument is NOT in _PLATFORM, so it must not appear
        source_with_extra = _SOURCE | {"haystack.components.converters.csv.CSVToDocument"}
        platform_without = {k: v for k, v in _PLATFORM.items() if "csv" not in k}
        mdx = build_mdx(platform_without, source_with_extra)
        assert "CSVToDocument" not in mdx

    def test_schema_component_not_in_source_excluded_from_table(self):
        # Component in schema but not in source scan must not appear in the table
        platform_extra = {
            **_PLATFORM,
            "haystack.components.foo.MissingComponent": {"title": "MissingComponent", "family": ""},
        }
        mdx = build_mdx(platform_extra, _SOURCE)
        assert "MissingComponent" not in mdx

    def test_core_components_section(self):
        mdx = build_mdx(_PLATFORM, _SOURCE)
        assert "## Core Components" in mdx
        assert "InMemoryBM25Retriever" in mdx

    def test_integration_partner_section(self):
        mdx = build_mdx(_PLATFORM, _SOURCE)
        assert "## OpenSearch" in mdx
        assert "OpenSearchBM25Retriever" in mdx

    def test_available_marker_present(self):
        mdx = build_mdx(_PLATFORM, _SOURCE)
        assert "✅ Available" in mdx

    def test_generator_components_excluded(self):
        platform = {
            "haystack.components.generators.openai.OpenAIGenerator": {
                "title": "OpenAIGenerator",
                "family": "generators",
            },
            "haystack.components.retrievers.memory.InMemoryBM25Retriever": {
                "title": "InMemoryBM25Retriever",
                "family": "retrievers",
            },
        }
        source = set(platform.keys())
        mdx = build_mdx(platform, source)
        assert "OpenAIGenerator" not in mdx
        assert "InMemoryBM25Retriever" in mdx

    def test_chat_generator_components_included(self):
        platform = {
            "haystack.components.generators.openai.OpenAIChatGenerator": {
                "title": "OpenAIChatGenerator",
                "family": "generators",
            },
            "haystack.components.generators.openai.OpenAIGenerator": {
                "title": "OpenAIGenerator",
                "family": "generators",
            },
        }
        source = set(platform.keys())
        mdx = build_mdx(platform, source)
        assert "OpenAIChatGenerator" in mdx
        assert "OpenAIGenerator" not in mdx

    def test_voyage_embedders_and_ranker_merged_under_voyage_ai(self):
        platform = {
            "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder.VoyageTextEmbedder": {
                "title": "VoyageTextEmbedder",
                "family": "embedders",
            },
            "haystack_integrations.components.rankers.voyage.ranker.VoyageRanker": {
                "title": "VoyageRanker",
                "family": "rankers",
            },
        }
        source = set(platform.keys())
        mdx = build_mdx(platform, source)
        # Must appear exactly once as a section heading
        assert mdx.count("## Voyage AI") == 1
        assert "VoyageTextEmbedder" in mdx
        assert "VoyageRanker" in mdx

    def test_docling_serve_merged_under_docling(self):
        platform = {
            "haystack_integrations.components.converters.docling.converter.DoclingConverter": {
                "title": "DoclingConverter",
                "family": "converters",
            },
            "haystack_integrations.components.converters.docling_serve.converter.DoclingServeConverter": {
                "title": "DoclingServeConverter",
                "family": "converters",
            },
        }
        source = set(platform.keys())
        mdx = build_mdx(platform, source)
        assert mdx.count("## Docling") == 1

    def test_family_field_used_for_type(self):
        platform = {
            "haystack.components.retrievers.memory.InMemoryBM25Retriever": {
                "title": "InMemoryBM25Retriever",
                "family": "retrievers",
            }
        }
        link_map = {"InMemoryBM25Retriever": "https://docs.haystack.deepset.ai/docs/inmemorybm25retriever"}
        mdx = build_mdx(platform, set(platform.keys()), docs_link_map=link_map)
        assert "| [InMemoryBM25Retriever](https://docs.haystack.deepset.ai/docs/inmemorybm25retriever) | Retriever |" in mdx

    def test_family_fallback_to_fqn_path(self):
        platform = {
            "haystack.components.rankers.foo.MyRanker": {
                "title": "MyRanker",
                "family": "",  # missing family
            }
        }
        mdx = build_mdx(platform, set(platform.keys()))
        assert "| MyRanker | Ranker |" in mdx

    def test_component_with_no_docs_page_is_plain_text(self):
        platform = {
            "haystack.components.rankers.foo.MyRanker": {
                "title": "MyRanker",
                "family": "rankers",
            }
        }
        # docs_link_map has no entry for MyRanker
        mdx = build_mdx(platform, set(platform.keys()), docs_link_map={})
        assert "| MyRanker | Ranker |" in mdx
        assert "[MyRanker]" not in mdx


# ── scan_docs_links / _parse_frontmatter ─────────────────────────────────────


class TestScanDocsLinks:
    def test_parses_title_and_slug(self, tmp_path):
        mdx = tmp_path / "pipeline-components" / "retrievers" / "inmemorybm25retriever.mdx"
        mdx.parent.mkdir(parents=True)
        mdx.write_text('---\ntitle: "InMemoryBM25Retriever"\nid: inmemorybm25retriever\nslug: "/inmemorybm25retriever"\n---\n')
        result = scan_docs_links(tmp_path)
        assert result == {"InMemoryBM25Retriever": "https://docs.haystack.deepset.ai/docs/inmemorybm25retriever"}

    def test_files_without_frontmatter_skipped(self, tmp_path):
        mdx = tmp_path / "no-frontmatter.mdx"
        mdx.write_text("# Just a heading\n")
        assert scan_docs_links(tmp_path) == {}

    def test_files_missing_slug_skipped(self, tmp_path):
        mdx = tmp_path / "partial.mdx"
        mdx.write_text('---\ntitle: "OnlyTitle"\n---\n')
        assert scan_docs_links(tmp_path) == {}

    def test_multiple_files_all_indexed(self, tmp_path):
        for name in ("foo", "bar"):
            f = tmp_path / f"{name}.mdx"
            f.write_text(f'---\ntitle: "{name.title()}"\nslug: "/{name}"\n---\n')
        result = scan_docs_links(tmp_path)
        assert len(result) == 2
        assert result["Foo"] == "https://docs.haystack.deepset.ai/docs/foo"


class TestParseFrontmatter:
    def test_basic_key_value(self):
        text = '---\ntitle: "Hello"\nslug: "/hello"\n---\nBody'
        fm = _parse_frontmatter(text)
        assert fm["title"] == "Hello"
        assert fm["slug"] == "/hello"

    def test_no_frontmatter_returns_empty(self):
        assert _parse_frontmatter("# No frontmatter") == {}

    def test_unclosed_frontmatter_returns_empty(self):
        assert _parse_frontmatter("---\ntitle: orphan\n") == {}
