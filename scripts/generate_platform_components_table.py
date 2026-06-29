# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate a Haystack Enterprise Platform Components MDX table.

Scans deepset-ai/haystack and deepset-ai/haystack-core-integrations for classes
decorated with @component, cross-references them against the platform component
schema (schema-full-component-list.json from deepset-ai/haystack-runtime), and
writes a formatted MDX page.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Maps schema `family` value / module path segment → display type name
TYPE_MAP: dict[str, str] = {
    "generators": "Generator",
    "retrievers": "Retriever",
    "embedders": "Embedder",
    "converters": "Converter",
    "rankers": "Ranker",
    "document_stores": "Document Store",
    "preprocessors": "Preprocessor",
    "readers": "Reader",
    "routers": "Router",
    "joiners": "Joiner",
    "builders": "Builder",
    "classifiers": "Classifier",
    "extractors": "Extractor",
    "samplers": "Sampler",
    "writers": "Writer",
    "connectors": "Connector",
    "fetchers": "Fetcher",
    "validators": "Validator",
    "audio": "Audio",
}

# Human-readable partner labels; snake_case partner keys → display name.
# Keys that map to the same value are intentionally merged into one section.
PARTNER_LABELS: dict[str, str] = {
    "aimlapi": "AIML API",
    "alloydb": "AlloyDB",
    "amazon_bedrock": "Amazon Bedrock",
    "amazon_sagemaker": "Amazon SageMaker",
    "amazon_textract": "Amazon Textract",
    "anthropic": "Anthropic",
    "arcadedb": "ArcadeDB",
    "astra": "Astra",
    "azure_ai_search": "Azure AI Search",
    "azure_doc_intelligence": "Azure Document Intelligence",
    "brave": "Brave",
    "chroma": "Chroma",
    "cohere": "Cohere",
    "docling": "Docling",
    "docling_serve": "Docling",  # merged with docling
    "elasticsearch": "Elasticsearch",
    "falkordb": "FalkorDB",
    "fastembed": "FastEmbed",
    "firecrawl": "Firecrawl",
    "github": "GitHub",
    "google_genai": "Google Generative AI",
    "huggingface_api": "Hugging Face API",
    "jina": "Jina AI",
    "langfuse": "Langfuse",
    "lara": "Lara",
    "litellm": "LiteLLM",
    "llama_cpp": "Llama.cpp",
    "llama_stack": "Llama Stack",
    "markitdown": "MarkItDown",
    "mem0": "Mem0",
    "meta_llama": "Meta Llama",
    "mistral": "Mistral",
    "mongodb_atlas": "MongoDB Atlas",
    "nvidia": "NVIDIA",
    "ollama": "Ollama",
    "opensearch": "OpenSearch",
    "openrouter": "OpenRouter",
    "oracle": "Oracle",
    "perplexity": "Perplexity",
    "pgvector": "pgvector",
    "pinecone": "Pinecone",
    "presidio": "Presidio",
    "pyversity": "Pyversity",
    "qdrant": "Qdrant",
    "s3": "Amazon S3",
    "snowflake": "Snowflake",
    "sqlalchemy": "SQLAlchemy",
    "stackit": "STACKIT",
    "supabase": "Supabase",
    "tavily": "Tavily",
    "togetherai": "TogetherAI",
    "unstructured": "Unstructured",
    "valkey": "Valkey",
    "vespa": "Vespa",
    "vllm": "vLLM",
    "voyage": "Voyage AI",
    "voyage_embedders": "Voyage AI",  # merged with voyage
    "weave": "Weights & Biases (Weave)",
    "weaviate": "Weaviate",
}

_HAYSTACK_DOCS_BASE = "https://docs.haystack.deepset.ai"
_HAYSTACK_DOCS_PREFIX = "/docs"  # Docusaurus routeBasePath

# Namespaces that must never appear in public documentation
_EXCLUDED_NAMESPACES = ("deepset_cloud_custom_nodes.", "studio_internal.", "haystack_experimental.", "deepl_haystack.")

# Individual class names to exclude regardless of namespace
_EXCLUDED_CLASS_NAMES = {
    "SuperComponent",  # internal base class, not a user-facing component
    "LLM",  # alias not present in Haystack OS
}

# Sets allow O(1) membership tests
_PACKAGE_ROOTS: set[str] = {"haystack_integrations", "haystack"}
_COMPONENT_DECORATORS: set[str] = {"component", "super_component"}


# ── Component type helpers ────────────────────────────────────────────────────


def infer_type(family: str | None, fqn: str) -> str:
    """Return a human-readable type string from the schema family or FQN path segments."""
    if family and family in TYPE_MAP:
        return TYPE_MAP[family]
    return next((TYPE_MAP[seg] for seg in fqn.split(".") if seg in TYPE_MAP), "Component")


def partner_label_for(fqn: str) -> str | None:
    """Return the display-name partner label for a ``haystack_integrations.*`` FQN."""
    parts = fqn.split(".")
    # haystack_integrations . components . <family> . <partner> . …
    if len(parts) < 4:
        return None
    key = parts[3]
    return PARTNER_LABELS.get(key, key.replace("_", " ").title())


# ── Docs-site link map ────────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract key-value pairs from YAML-style frontmatter delimited by ``---``."""
    if not text.startswith("---"):
        return {}
    try:
        end = text.index("---", 3)
    except ValueError:
        return {}
    result: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip('"')
    return result


def scan_docs_links(docs_src: Path) -> dict[str, str]:
    """
    Scan docs-website MDX files and return ``{component_title: absolute_url}``.

    Reads only the first 1 KB of each file — enough to cover the frontmatter —
    avoiding the cost of loading large MDX files in full.
    """
    link_map: dict[str, str] = {}
    for mdx_file in docs_src.rglob("*.mdx"):
        try:
            with mdx_file.open(encoding="utf-8") as fh:
                head = fh.read(1024)
        except OSError:
            continue
        fm = _parse_frontmatter(head)
        title = fm.get("title", "")
        slug = fm.get("slug", "")
        if title and slug:
            link_map[title] = f"{_HAYSTACK_DOCS_BASE}{_HAYSTACK_DOCS_PREFIX}{slug}"
    return link_map


# ── Source-scanning helpers ───────────────────────────────────────────────────


def has_component_decorator(node: ast.ClassDef) -> bool:
    """Return True if *node* has a ``@component`` or ``@super_component`` decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in _COMPONENT_DECORATORS:
            return True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id in _COMPONENT_DECORATORS:
            return True
    return False


def compute_module_path(file_path: Path, repo_root: Path) -> str | None:
    """Convert a ``.py`` file path to a dotted Python module path."""
    try:
        parts = list(file_path.relative_to(repo_root).with_suffix("").parts)
    except ValueError:
        return None
    if parts and parts[-1] == "__init__":
        parts.pop()
    start = next((i for i, p in enumerate(parts) if p in _PACKAGE_ROOTS), None)
    return ".".join(parts[start:]) if start is not None else None


def scan_for_components(scan_dir: Path, repo_root: Path) -> set[str]:
    """Scan *scan_dir* for Python files with ``@component``-decorated classes."""
    found: set[str] = set()
    py_files = list(scan_dir.rglob("*.py"))
    logger.info("Scanned %d Python files in %s", len(py_files), scan_dir)

    for py_file in py_files:
        module_path = compute_module_path(py_file, repo_root)
        if module_path is None:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # Fast pre-filter: skip files with no decorator mention at all
        if "@component" not in source and "@super_component" not in source:
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            logger.warning("Could not parse %s: %s", py_file, exc)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and has_component_decorator(node):
                found.add(f"{module_path}.{node.name}")

    return found


# ── Schema loading ────────────────────────────────────────────────────────────


def load_platform_components(schema_path: Path) -> dict[str, dict]:
    """Parse ``schema-full-component-list.json`` and return public components."""
    all_defs: dict = json.loads(schema_path.read_text(encoding="utf-8")).get("definitions", {}).get("Components", {})
    result: dict[str, dict] = {}
    for fqn, defn in all_defs.items():
        if any(fqn.startswith(ns) for ns in _EXCLUDED_NAMESPACES):
            continue
        if not fqn.startswith(("haystack.", "haystack_integrations.")):
            continue
        if fqn.split(".")[-1] in _EXCLUDED_CLASS_NAMES:
            continue
        type_props = defn.get("properties", {}).get("type", {})
        result[fqn] = {"title": defn.get("title", fqn.split(".")[-1]), "family": type_props.get("family", "")}
    return result


# ── MDX generation ────────────────────────────────────────────────────────────


def build_mdx(
    platform_components: dict[str, dict], source_components: set[str], docs_link_map: dict[str, str] | None = None
) -> str:
    """Render the MDX page content."""
    _link_map = docs_link_map or {}
    core_fqns: list[str] = []
    partner_components: dict[str, list[str]] = {}
    visible_count = 0

    for fqn in platform_components:
        if fqn not in source_components:
            logger.warning("Schema component not found in source scan: %s", fqn)
            continue
        cls = fqn.split(".")[-1]
        if cls.endswith("Generator") and not cls.endswith("ChatGenerator"):
            continue
        visible_count += 1
        if fqn.startswith("haystack_integrations."):
            partner_components.setdefault(partner_label_for(fqn) or "Other", []).append(fqn)
        else:
            core_fqns.append(fqn)

    def render_table(fqns: list[str]) -> list[str]:
        rows = [
            "| Component | Type | Haystack Enterprise Platform |",
            "|-----------|------|------------------------------|",
        ]
        for fqn in sorted(fqns, key=lambda x: x.split(".")[-1].lower()):
            meta = platform_components[fqn]
            name = meta["title"]
            link = _link_map.get(name, "")
            name_cell = f"[{name}]({link})" if link else name
            rows.append(f"| {name_cell} | {infer_type(meta.get('family'), fqn)} | ✅ Available |")
        return rows

    sections: list[str] = [
        "---",
        'title: "Haystack Enterprise Components"',
        "id: platform-components",
        'slug: "/platform-components"',
        'description: "A complete list of Haystack components available on the Haystack Enterprise Platform,'
        ' grouped by integration partner."',
        "---",
        "",
        "# Haystack Enterprise Components",
        "",
        f"The Haystack Enterprise Platform currently supports **{visible_count} components**"
        f" and **{len(partner_components)} integrations**."
        " The following table lists them grouped by integration partner.",
        "",
    ]

    if core_fqns:
        sections += ["## Core Components", "", *render_table(core_fqns), ""]

    for label in sorted(partner_components, key=str.lower):
        sections += [f"## {label}", "", *render_table(partner_components[label]), ""]

    return "\n".join(sections)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse arguments, run the scan, and write the MDX file."""
    parser = argparse.ArgumentParser(description="Generate the Haystack Enterprise Platform Components MDX table.")
    parser.add_argument(
        "--haystack-src", required=True, type=Path, metavar="PATH", help="Root of the deepset-ai/haystack checkout."
    )
    parser.add_argument(
        "--integrations-src",
        required=True,
        type=Path,
        metavar="PATH",
        help="Root of the deepset-ai/haystack-core-integrations checkout.",
    )
    parser.add_argument(
        "--schema", required=True, type=Path, metavar="PATH", help="Path to schema JSON file in the platform repo."
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        default=Path("docs-website/docs/overview/platform-components.mdx"),
        help="Destination .mdx file path (default: docs-website/docs/overview/platform-components.mdx).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated content to stdout instead of writing the file."
    )

    args = parser.parse_args(argv)

    for path, flag in [
        (args.haystack_src, "--haystack-src"),
        (args.integrations_src, "--integrations-src"),
        (args.schema, "--schema"),
    ]:
        if not path.exists():
            print(f"ERROR: {flag} path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        platform_components = load_platform_components(args.schema)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot load schema file: {exc}", file=sys.stderr)
        sys.exit(1)

    haystack_pkg = args.haystack_src / "haystack"
    source_components = scan_for_components(
        haystack_pkg if haystack_pkg.is_dir() else args.haystack_src, args.haystack_src
    )
    integrations_pkg = args.integrations_src / "integrations"
    source_components |= scan_for_components(
        integrations_pkg if integrations_pkg.is_dir() else args.integrations_src, args.integrations_src
    )

    logger.info("Platform components in schema (public namespaces): %d", len(platform_components))
    logger.info("Components matched to source scan: %d", sum(1 for f in platform_components if f in source_components))

    docs_src = args.haystack_src / "docs-website" / "docs"
    if not docs_src.is_dir():
        logger.warning("docs-website/docs not found under --haystack-src, component links will be omitted")
    docs_link_map = scan_docs_links(docs_src) if docs_src.is_dir() else {}

    mdx = build_mdx(platform_components, source_components, docs_link_map)

    if args.dry_run:
        print(mdx)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(mdx, encoding="utf-8")
        logger.info("Written to %s", args.output)


if __name__ == "__main__":
    main()
