import sys
import io
import dataclasses
import docspec
import typing as t

from pydoc_markdown.interfaces import Context, Renderer, Resolver
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


class HaystackMarkdownRenderer(MarkdownRenderer):
    """
    Custom Markdown renderer heavily based on the `MarkdownRenderer`,
    it shows the import path for classes when present.

    This only works in cooperation with a Processor that stores a
    field called `import_path` into the objects
    """

    def _render_header(self, fp, level, obj):
        super()._render_header(fp, level, obj)
        if hasattr(obj, "import_path"):
            fp.write(obj.import_path)
            fp.write("\n\n")


@dataclasses.dataclass
class ReadmeRenderer(Renderer):
    """
    This custom Renderer prepends a front matter so that the output can be published
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
    markdown: HaystackMarkdownRenderer = dataclasses.field(default_factory=HaystackMarkdownRenderer)

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

    def process(self, modules: t.List[docspec.Module], resolver: t.Optional[Resolver]) -> None:
        for mod in modules:
            for m in mod.members:
                if type(m) == docspec.Class:
                    # FIXME: we should compute the real import path here
                    m.import_path = "foo.bar.baz"
        self.markdown.process(modules, resolver)

    def _frontmatter(self) -> str:
        return README_FRONTMATTER.format(
            title=self.title, category=self.category, excerpt=self.excerpt, slug=self.slug, order=self.order
        )
