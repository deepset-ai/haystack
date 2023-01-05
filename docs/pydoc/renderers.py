import sys
import io
import dataclasses
import docspec
import typing as t
from pathlib import Path
from pydoc_markdown.interfaces import Context, Renderer, Resolver
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
import html

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
    Custom Markdown renderer heavily based on the `MarkdownRenderer`
    """
    def _render_object(self, fp, level, obj):
        """
        This is where docstrings for a certain object are processed,
        we need to override it in order to better manage new lines.
        """
        if not isinstance(obj, docspec.Module) or self.render_module_header:
            self._render_header(fp, level, obj)

        render_view_source = not isinstance(obj, (docspec.Module, docspec.Variable))

        if render_view_source:
            url = self.source_linker.get_source_url(obj) if self.source_linker else None
            source_string = self.source_format.replace("{url}", str(url)) if url else None
            if source_string and self.source_position == "before signature":
                fp.write(source_string + "\n\n")

        self._render_signature_block(fp, obj)

        if render_view_source:
            if source_string and self.source_position == "after signature":
                fp.write(source_string + "\n\n")

        if obj.docstring:
            docstring = html.escape(obj.docstring.content) if self.escape_html_in_docstring else obj.docstring.content
            docstring = docstring.replace("**Arguments**", "\n\n**Arguments**")
            docstring = docstring.replace("**Returns**", "\n\n**Returns**")
            docstring = docstring.replace("**Raises**", "\n\n**Raises**")
            lines = docstring.split("\n\n")
            if self.docstrings_as_blockquote:
                lines = ["> " + x for x in lines]
            fp.write("\n".join(lines))
            fp.write("\n\n")

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
    markdown: HaystackMarkdownRenderer = dataclasses.field(default_factory=HaystackMarkdownRenderer)

    def init(self, context: Context) -> None:
        self.markdown.init(context)

    def render(self, modules: t.List[docspec.Module]) -> None:
        if self.markdown.filename is None:
            sys.stdout.write(self._frontmatter())
            self.markdown._render_to_stream(modules, sys.stdout)
        else:
            with io.open(self.markdown.filename, "w", encoding=self.markdown.encoding) as fp:
                fp.write(self._frontmatter())
                self.markdown._render_to_stream(modules, t.cast(t.TextIO, fp))

    def _frontmatter(self) -> str:
        return README_FRONTMATTER.format(
            title=self.title, category=self.category, excerpt=self.excerpt, slug=self.slug, order=self.order
        )