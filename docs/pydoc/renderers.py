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
---

"""


@dataclasses.dataclass
class ReadmeRenderer(Renderer):

    title: str
    category: str
    excerpt: str

    markdown: MarkdownRenderer = dataclasses.field(default_factory=MarkdownRenderer)

    def init(self, context: Context) -> None:
        self.markdown.init(context)

    def render(self, modules: t.List[docspec.Module]) -> None:
        if self.markdown.filename is None:
            sys.stdout.write(self._frontmatter())
            self.markdown.render_to_stream(modules, sys.stdout)
        else:
            with io.open(self.markdown.filename, "w", encoding=self.markdown.encoding) as fp:
                fp.write(self._frontmatter())
                self.markdown.render_to_stream(modules, t.cast(t.TextIO, fp))

    def _frontmatter(self) -> str:
        return README_FRONTMATTER.format(title=self.title, category=self.category, excerpt=self.excerpt)
