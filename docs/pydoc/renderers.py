import sys
import io
import dataclasses
import docspec
import typing as t
from pathlib import Path
from pydoc_markdown.interfaces import Context, Renderer
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer


README_FRONTMATTER = """---
title: {title}
excerpt: {excerpt}
category: {category}
slug: {slug}
order: {order}
hidden: false
---

"""


@dataclasses.dataclass
class ReadmeRenderer(Renderer):
    """
    This custom Renderer is heavily based on the `MarkdownRenderer`,
    it just prepends a front matter so that the output can be published
    directly to readme.io.
    """

    # These settings will be used in the front matter output
    title: str
    category: str
    excerpt: str
    slug: str
    order: int
    # This exposes a special `markdown` settings value that can be used to pass
    # parameters to the underlying `MarkdownRenderer`
    markdown: MarkdownRenderer = dataclasses.field(default_factory=MarkdownRenderer)

    def init(self, context: Context) -> None:
        self.markdown.init(context)

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
            title=self.title, category=self.category, excerpt=self.excerpt, slug=self.slug, order=self.order
        )
