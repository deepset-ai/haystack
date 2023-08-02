import os
import sys
import io
import dataclasses
import typing as t
import base64
import warnings
from pathlib import Path

import requests
import docspec
from pydoc_markdown.interfaces import Context, Renderer
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer


README_FRONTMATTER = """---
title: {title}
excerpt: {excerpt}
category: {category}
slug: {slug}
parentDoc: {parent_doc}
order: {order}
hidden: false
---

"""


def create_headers(version: str):
    # Utility function to create Readme.io headers.
    # We assume the README_API_KEY env var is set since we check outside
    # to show clearer error messages.
    README_API_KEY = os.getenv("README_API_KEY")
    token = base64.b64encode(f"{README_API_KEY}:".encode()).decode()
    return {"authorization": f"Basic {token}", "x-readme-version": version}


@dataclasses.dataclass
class ReadmeRenderer(Renderer):
    """
    This custom Renderer is heavily based on the `MarkdownRenderer`,
    it just prepends a front matter so that the output can be published
    directly to readme.io.
    """

    # These settings will be used in the front matter output
    title: str
    category_slug: str
    excerpt: str
    slug: str
    order: int
    parent_doc_slug: str = ""
    # Docs categories fetched from Readme.io
    categories: t.Dict[str, str] = dataclasses.field(init=False)
    # This exposes a special `markdown` settings value that can be used to pass
    # parameters to the underlying `MarkdownRenderer`
    markdown: MarkdownRenderer = dataclasses.field(default_factory=MarkdownRenderer)

    def init(self, context: Context) -> None:
        self.markdown.init(context)
        self.version = self._doc_version()
        self.categories = self._readme_categories(self.version)

    def _doc_version(self) -> str:
        """
        Returns the docs version.
        """
        root = Path(__file__).absolute().parent.parent.parent
        full_version = (root / "VERSION.txt").read_text()
        major, minor = full_version.split(".")[:2]
        if "rc0" in full_version:
            return f"v{major}.{minor}-unstable"
        return f"v{major}.{minor}"

    def _readme_categories(self, version: str) -> t.Dict[str, str]:
        """
        Fetch the categories of the given version from Readme.io.
        README_API_KEY env var must be set to correctly get the categories.
        Returns dictionary containing all the categories slugs and their ids.
        """
        README_API_KEY = os.getenv("README_API_KEY")
        if not README_API_KEY:
            warnings.warn("README_API_KEY env var is not set, using a placeholder category ID")
            return {"haystack-classes": "ID"}

        headers = create_headers(version)

        res = requests.get("https://dash.readme.com/api/v1/categories", headers=headers, timeout=60)

        if not res.ok:
            sys.exit(f"Error requesting {version} categories")

        return {c["slug"]: c["id"] for c in res.json()}

    def _doc_id(self, doc_slug: str, version: str) -> str:
        """
        Fetch the doc id of the given doc slug and version from Readme.io.
        README_API_KEY env var must be set to correctly get the id.
        If doc_slug is an empty string return an empty string.
        """
        if not doc_slug:
            # Not all docs have a parent doc, in case we get no slug
            # we just return an empty string.
            return ""

        README_API_KEY = os.getenv("README_API_KEY")
        if not README_API_KEY:
            warnings.warn("README_API_KEY env var is not set, using a placeholder doc ID")
            return "fake-doc-id"

        headers = create_headers(version)
        res = requests.get(f"https://dash.readme.com/api/v1/docs/{doc_slug}", headers=headers, timeout=60)
        if not res.ok:
            sys.exit(f"Error requesting {doc_slug} doc for version {version}")

        return res.json()["id"]

    def render(self, modules: t.List[docspec.Module]) -> None:
        if self.markdown.filename is None:
            sys.stdout.write(self._frontmatter())
            self.markdown.render_single_page(sys.stdout, modules)
        else:
            with io.open(self.markdown.filename, "w", encoding=self.markdown.encoding) as fp:
                fp.write(self._frontmatter())
                self.markdown.render_single_page(t.cast(t.TextIO, fp), modules)

    def _frontmatter(self) -> str:
        return README_FRONTMATTER.format(
            title=self.title,
            category=self.categories[self.category_slug],
            parent_doc=self._doc_id(self.parent_doc_slug, self.version),
            excerpt=self.excerpt,
            slug=self.slug,
            order=self.order,
        )
