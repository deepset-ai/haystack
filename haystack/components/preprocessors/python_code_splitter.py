# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import math
from dataclasses import dataclass, field
from typing import Any

from haystack import Document, component, logging
from haystack.components.preprocessors.document_splitter import DocumentSplitter

logger = logging.getLogger(__name__)


@dataclass
class _CodeUnit:
    """One syntactic split unit (function, class header, method, imports block, statement, ...)."""

    source: str
    start_line: int
    end_line: int
    kind: str
    name: str | None = None
    class_name: str | None = None
    class_signature: str | None = None
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None


@component
class PythonCodeSplitter:
    """
    Split Python source code into syntax-aware chunks.

    The component parses each source with :mod:`ast` into *units* (module docstring,
    consecutive ``import`` blocks, top-level functions, class headers, methods, nested
    classes, and remaining statements) and merges them greedily in source order toward
    ``max_effective_lines`` per chunk, where effective lines are
    ``ceil(len(source) / expected_chars_per_line)``. Functions and methods are kept
    whole; the resulting chunks read top-to-bottom like the original file with comments
    and blank lines preserved.

    A function whose effective length exceeds ``oversized_factor * max_effective_lines``
    is the only case where chunks may overlap: it is broken down with a line-based
    secondary split (:class:`DocumentSplitter`, ``split_by="line"``) and the resulting
    pieces carry ``secondary_split=True`` along with the originating function's metadata.
    The primary split never adds overlap.

    Per-chunk metadata: ``source_id``, ``split_id``, ``start_line``, ``end_line``,
    ``unit_kinds``; plus ``include_classes``, ``decorators``, and ``docstrings`` (when
    ``strip_docstrings=True``) where applicable. ``file_name`` and any other parent
    document meta are propagated.

    Usage example:

    ```python
    from haystack import Document
    from haystack.components.preprocessors import PythonCodeSplitter

    source = '''
    \"\"\"Example module.\"\"\"
    from math import sqrt


    class Circle:
        def __init__(self, r: float) -> None:
            self.r = r

        def area(self) -> float:
            return 3.14159 * self.r * self.r
    '''

    splitter = PythonCodeSplitter(min_effective_lines=4, max_effective_lines=6)
    result = splitter.run(documents=[Document(content=source, meta={"file_name": "circle.py"})])
    for chunk in result["documents"]:
        print(chunk.meta["start_line"], chunk.meta["end_line"], chunk.meta.get("include_classes"))
    ```

    Pass ``strip_docstrings=True`` to move docstrings out of the chunk content and into
    each chunk's ``meta["docstrings"]`` list. This is useful for RAG when docstrings are
    large: stripping shrinks the stored content while the docstring text can still
    influence retrieval via ``meta_fields_to_embed=["docstrings"]`` on the embedder.
    """

    def __init__(
        self,
        *,
        min_effective_lines: int = 20,
        max_effective_lines: int = 100,
        expected_chars_per_line: int = 45,
        oversized_factor: int = 3,
        strip_docstrings: bool = False,
        preserve_class_definition: bool = True,
        secondary_split_overlap: int = 5,
        secondary_split_length: int | None = None,
    ) -> None:
        """
        Initialize the PythonCodeSplitter.

        :param min_effective_lines: Minimum effective lines per chunk. While the running
            chunk is below this threshold the splitter keeps merging in the next unit.
        :param max_effective_lines: Target effective lines per chunk. Units are merged
            greedily while doing so brings the running total closer to this target.
        :param expected_chars_per_line: Used to convert characters into effective lines as
            ``ceil(len(source) / expected_chars_per_line)``; long lines count as more than one.
        :param oversized_factor: A function whose effective length exceeds
            ``oversized_factor * max_effective_lines`` triggers the line-based secondary
            split with overlap.
        :param strip_docstrings: If ``True``, function/method/class docstrings are moved
            from the chunk content into ``meta["docstrings"]`` (source order). The
            module-level docstring is kept in place since it is itself a top-level unit.
        :param preserve_class_definition: If ``True`` (default), chunks that contain class
            members but not the class header are prefixed with the bare class signature
            (decorators plus the ``class Foo(...):`` lines) in source order.
        :param secondary_split_overlap: Line overlap for the secondary splitter; only used
            in the oversized fallback. The primary AST split never adds overlap.
        :param secondary_split_length: Lines per chunk for the secondary splitter.
            Defaults to ``max_effective_lines`` when ``None``.
        :raises ValueError: If any parameter is invalid (negative, zero where positive is
            required, or ``min_effective_lines > max_effective_lines``).
        """
        if min_effective_lines < 1:
            raise ValueError("min_effective_lines must be at least 1.")
        if max_effective_lines < 1:
            raise ValueError("max_effective_lines must be at least 1.")
        if min_effective_lines > max_effective_lines:
            raise ValueError("min_effective_lines must not be greater than max_effective_lines.")
        if expected_chars_per_line < 1:
            raise ValueError("expected_chars_per_line must be at least 1.")
        if oversized_factor < 1:
            raise ValueError("oversized_factor must be at least 1.")
        if secondary_split_overlap < 0:
            raise ValueError("secondary_split_overlap must be non-negative.")
        if secondary_split_length is not None and secondary_split_length < 1:
            raise ValueError("secondary_split_length must be at least 1.")

        self.min_effective_lines = min_effective_lines
        self.max_effective_lines = max_effective_lines
        self.expected_chars_per_line = expected_chars_per_line
        self.oversized_factor = oversized_factor
        self.strip_docstrings = strip_docstrings
        self.preserve_class_definition = preserve_class_definition
        self.secondary_split_overlap = secondary_split_overlap
        self.secondary_split_length = secondary_split_length

    def _effective_lines(self, text: str) -> int:
        """Return the number of *effective lines* for ``text`` (see class docstring)."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.expected_chars_per_line))

    def _is_oversized(self, unit: "_CodeUnit") -> bool:
        """Return ``True`` if ``unit`` should trigger the secondary line-based split."""
        return self._effective_lines(unit.source) > self.oversized_factor * self.max_effective_lines

    @staticmethod
    def _slice_lines(source_lines: list[str], start: int, end: int) -> str:
        """Slice ``source_lines`` between the 1-indexed ``start`` and ``end`` (inclusive)."""
        start = max(start, 1)
        if end < start:
            return ""
        return "".join(source_lines[start - 1 : end])

    @staticmethod
    def _safe_unparse(node: ast.AST) -> str:
        """Return ``ast.unparse(node)`` but tolerate exotic nodes by falling back to ``repr``."""
        try:
            return ast.unparse(node)
        except Exception:  # pragma: no cover - defensive guard
            return repr(node)

    def _strip_docstring(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        source_lines: list[str],
        unit_start: int,
        unit_end: int,
    ) -> tuple[str, str | None]:
        """Strip ``node``'s docstring from ``source_lines[unit_start..unit_end]`` if safely possible."""
        docstring = ast.get_docstring(node)
        body = node.body
        if not docstring or not body:
            return self._slice_lines(source_lines, unit_start, unit_end), None

        first = body[0]
        if not (
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        ):
            return self._slice_lines(source_lines, unit_start, unit_end), None

        # Skip stripping when the docstring shares a line with the def/class (would
        # leave broken syntax) or extends past the caller's slice (e.g. class_header).
        ds_start = first.lineno
        ds_end = first.end_lineno or first.lineno
        if ds_start <= node.lineno or ds_end > unit_end:
            return self._slice_lines(source_lines, unit_start, unit_end), None

        before = source_lines[unit_start - 1 : ds_start - 1]
        after = source_lines[ds_end:unit_end]
        return "".join(before + after), docstring

    def _emit_class_units(self, cls: ast.ClassDef, source_lines: list[str], cursor: int, units: list[_CodeUnit]) -> int:
        """Emit class header and per-method units for ``cls``; return the next cursor (1-indexed)."""
        class_start = cls.decorator_list[0].lineno if cls.decorator_list else cls.lineno
        class_end = cls.end_lineno or cls.lineno
        class_name = cls.name
        class_decorators = [self._safe_unparse(d) for d in cls.decorator_list]

        # Methods, async methods, and nested classes become their own units so a
        # method is never split mid-statement.
        split_children_idx = [
            k
            for k, child in enumerate(cls.body)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        # Bare class signature (decorators + ``class Foo(...):`` lines) used by
        # ``preserve_class_definition`` to prefix later chunks of the same class.
        class_signature: str | None = None
        if cls.body:
            body_start = cls.body[0].lineno
            if body_start > class_start:
                class_signature = self._slice_lines(source_lines, class_start, body_start - 1)

        # Whole class fits in one unit when there are no inner split points.
        if not split_children_idx:
            unit_slice = self._slice_lines(source_lines, cursor, class_end)
            stripped_docstring: str | None = None
            if self.strip_docstrings:
                unit_slice, stripped_docstring = self._strip_docstring(cls, source_lines, cursor, class_end)
            units.append(
                _CodeUnit(
                    source=unit_slice,
                    start_line=class_start,
                    end_line=class_end,
                    kind="class",
                    name=class_name,
                    class_name=class_name,
                    class_signature=class_signature,
                    decorators=class_decorators,
                    docstring=stripped_docstring,
                )
            )
            return class_end + 1

        # Class header: from outer cursor up to (but excluding) the first split child.
        first_child = cls.body[split_children_idx[0]]
        if (
            isinstance(first_child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and first_child.decorator_list
        ):
            first_child_start = first_child.decorator_list[0].lineno
        else:
            first_child_start = first_child.lineno
        header_end = first_child_start - 1
        header_slice = self._slice_lines(source_lines, cursor, header_end)
        header_docstring: str | None = None
        if self.strip_docstrings:
            header_slice, header_docstring = self._strip_docstring(cls, source_lines, cursor, header_end)

        units.append(
            _CodeUnit(
                source=header_slice,
                start_line=class_start,
                end_line=header_end,
                kind="class_header",
                name=class_name,
                class_name=class_name,
                class_signature=class_signature,
                decorators=class_decorators,
                docstring=header_docstring,
            )
        )
        inner_cursor = header_end + 1

        for idx in split_children_idx:
            child = cls.body[idx]
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # narrowed above; kept for the type checker
            child_start = child.decorator_list[0].lineno if child.decorator_list else child.lineno
            child_end = child.end_lineno or child.lineno
            decorators = [self._safe_unparse(d) for d in child.decorator_list]

            unit_slice = self._slice_lines(source_lines, inner_cursor, child_end)
            stripped_docstring = None
            if self.strip_docstrings:
                unit_slice, stripped_docstring = self._strip_docstring(child, source_lines, inner_cursor, child_end)

            kind = "method" if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else "nested_class"
            units.append(
                _CodeUnit(
                    source=unit_slice,
                    start_line=child_start,
                    end_line=child_end,
                    kind=kind,
                    name=child.name,
                    class_name=class_name,
                    class_signature=class_signature,
                    decorators=decorators,
                    docstring=stripped_docstring,
                )
            )
            inner_cursor = child_end + 1

        # Append trailing class-body lines (comments / blanks after the last method).
        if inner_cursor <= class_end and units:
            trailing = self._slice_lines(source_lines, inner_cursor, class_end)
            units[-1].source += trailing
            units[-1].end_line = class_end

        return class_end + 1

    def _extract_units(self, source: str) -> list[_CodeUnit]:
        """Parse ``source`` and produce the ordered list of syntactic split units."""
        tree = ast.parse(source)
        source_lines = source.splitlines(keepends=True)
        total_lines = len(source_lines)

        units: list[_CodeUnit] = []
        cursor = 1
        body = tree.body
        node_idx = 0
        node_count = len(body)

        while node_idx < node_count:
            node = body[node_idx]

            # Module docstring (only valid as the very first statement).
            if (
                node_idx == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                end = node.end_lineno or node.lineno
                units.append(
                    _CodeUnit(
                        source=self._slice_lines(source_lines, cursor, end),
                        start_line=node.lineno,
                        end_line=end,
                        kind="module_docstring",
                    )
                )
                cursor = end + 1
                node_idx += 1
                continue

            # Group consecutive imports into one unit.
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_end_idx = node_idx
                while import_end_idx < node_count and isinstance(body[import_end_idx], (ast.Import, ast.ImportFrom)):
                    import_end_idx += 1
                last = body[import_end_idx - 1]
                end = last.end_lineno or last.lineno
                units.append(
                    _CodeUnit(
                        source=self._slice_lines(source_lines, cursor, end),
                        start_line=node.lineno,
                        end_line=end,
                        kind="imports",
                    )
                )
                cursor = end + 1
                node_idx = import_end_idx
                continue

            if isinstance(node, ast.ClassDef):
                cursor = self._emit_class_units(node, source_lines, cursor, units)
                node_idx += 1
                continue

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                end = node.end_lineno or node.lineno
                decorators = [self._safe_unparse(d) for d in node.decorator_list]
                unit_slice = self._slice_lines(source_lines, cursor, end)
                stripped_docstring: str | None = None
                if self.strip_docstrings:
                    unit_slice, stripped_docstring = self._strip_docstring(node, source_lines, cursor, end)
                units.append(
                    _CodeUnit(
                        source=unit_slice,
                        start_line=start,
                        end_line=end,
                        kind="function",
                        name=node.name,
                        decorators=decorators,
                        docstring=stripped_docstring,
                    )
                )
                cursor = end + 1
                node_idx += 1
                continue

            # Catch-all for top-level statements (assignments, conditionals, etc.).
            end = node.end_lineno or node.lineno
            units.append(
                _CodeUnit(
                    source=self._slice_lines(source_lines, cursor, end),
                    start_line=node.lineno,
                    end_line=end,
                    kind="statement",
                )
            )
            cursor = end + 1
            node_idx += 1

        # Append trailing content (comments after the last node) so the split is loss-less.
        if cursor <= total_lines and units:
            trailing = self._slice_lines(source_lines, cursor, total_lines)
            units[-1].source += trailing
            units[-1].end_line = total_lines
        elif cursor <= total_lines and not units:
            units.append(
                _CodeUnit(
                    source=self._slice_lines(source_lines, cursor, total_lines),
                    start_line=cursor,
                    end_line=total_lines,
                    kind="statement",
                )
            )

        return units

    def _merge_units(self, units: list[_CodeUnit]) -> list[list[_CodeUnit]]:
        """Greedily merge units toward ``max_effective_lines``; oversized units become solo chunks."""
        chunks: list[list[_CodeUnit]] = []
        current: list[_CodeUnit] = []
        current_lines = 0
        target = self.max_effective_lines

        def flush() -> None:
            nonlocal current, current_lines
            if current:
                chunks.append(current)
                current = []
                current_lines = 0

        for unit in units:
            if self._is_oversized(unit):
                flush()
                chunks.append([unit])
                continue

            unit_eff = self._effective_lines(unit.source)

            if not current:
                current = [unit]
                current_lines = unit_eff
                continue

            # Keep merging while below the minimum or while adding moves us closer to the target.
            new_total = current_lines + unit_eff
            if current_lines < self.min_effective_lines or abs(new_total - target) < abs(current_lines - target):
                current.append(unit)
                current_lines = new_total
            else:
                flush()
                current = [unit]
                current_lines = unit_eff

        flush()
        return chunks

    @staticmethod
    def _ordered_unique(items: list[str]) -> list[str]:
        """Return the list of unique items in their first-seen order."""
        return list(dict.fromkeys(items))

    def _build_chunk_meta(self, chunk: list[_CodeUnit], parent_doc: Document) -> dict[str, Any]:
        """Construct the output meta dict for a chunk of merged units."""
        meta: dict[str, Any] = {}
        if parent_doc.meta:
            meta.update({k: v for k, v in parent_doc.meta.items() if k not in {"split_id"}})
        meta["source_id"] = parent_doc.id

        # Units are emitted in source order, so chunk[0]/chunk[-1] give the extremes.
        meta["start_line"] = chunk[0].start_line
        meta["end_line"] = chunk[-1].end_line
        meta["unit_kinds"] = [u.kind for u in chunk]

        include_classes = self._ordered_unique([u.class_name for u in chunk if u.class_name])
        if include_classes:
            meta["include_classes"] = include_classes

        decorators: list[str] = []
        for u in chunk:
            decorators.extend(u.decorators)
        decorators = self._ordered_unique(decorators)
        if decorators:
            meta["decorators"] = decorators

        if self.strip_docstrings:
            docstrings = [u.docstring for u in chunk if u.docstring]
            if docstrings:
                meta["docstrings"] = docstrings

        return meta

    def _render_chunk_content(self, chunk: list[_CodeUnit]) -> str:
        """Render chunk content, optionally prefixing class signatures for orphan members."""
        body = "".join(u.source for u in chunk)
        if not self.preserve_class_definition:
            return body

        classes_with_header = {u.class_name for u in chunk if u.kind in {"class", "class_header"} and u.class_name}
        prepended: list[str] = []
        seen: set[str] = set()
        for u in chunk:
            if (
                u.class_name
                and u.class_name not in classes_with_header
                and u.class_name not in seen
                and u.class_signature
            ):
                prepended.append(u.class_signature)
                seen.add(u.class_name)

        if not prepended:
            return body
        return "".join(prepended) + body

    def _secondary_split(self, unit: _CodeUnit, parent_doc: Document) -> list[Document]:
        """Apply a line-based fallback split with overlap to a single oversized unit."""
        qualified_name = unit.name or unit.kind
        if unit.class_name and unit.name:
            qualified_name = f"{unit.class_name}.{unit.name}"
        logger.warning(
            "Oversized {kind} '{func_name}' at lines {start}-{end} ({eff} effective lines) exceeds "
            "{factor}x max_effective_lines={max_effective_lines}; falling back to line-based secondary split "
            "with overlap={overlap}.",
            kind=unit.kind,
            func_name=qualified_name,
            start=unit.start_line,
            end=unit.end_line,
            eff=self._effective_lines(unit.source),
            factor=self.oversized_factor,
            max_effective_lines=self.max_effective_lines,
            overlap=self.secondary_split_overlap,
        )

        # DocumentSplitter measures in physical lines; this approximates effective lines.
        split_length = (
            self.secondary_split_length if self.secondary_split_length is not None else self.max_effective_lines
        )
        overlap = min(self.secondary_split_overlap, max(0, split_length - 1))

        splitter = DocumentSplitter(split_by="line", split_length=split_length, split_overlap=overlap)
        intermediate = splitter.run(documents=[Document(content=unit.source)])["documents"]

        base_meta = self._build_chunk_meta([unit], parent_doc)
        results: list[Document] = []
        for idx, piece in enumerate(intermediate):
            meta = dict(base_meta)
            meta["secondary_split"] = True
            meta["secondary_split_index"] = idx
            meta["secondary_split_total"] = len(intermediate)
            results.append(Document(content=piece.content or "", meta=meta))
        return results

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Split each Python source ``Document`` into syntax-aware chunks.

        :param documents: Documents whose ``content`` is Python source code. Each
            document's ``meta`` is propagated onto its chunks.
        :returns: ``{"documents": [...]}`` where each chunk's meta additionally carries
            ``source_id``, ``split_id``, ``start_line``, ``end_line``, ``unit_kinds`` and
            - where applicable - ``include_classes``, ``decorators``, ``docstrings``,
            ``secondary_split``.
        :raises ValueError: If any document's content is ``None``.
        :raises TypeError: If any document's content is not a string.
        :raises SyntaxError: If a document's content is not valid Python.
        """
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"PythonCodeSplitter only works with text documents but content for document ID {doc.id} is None."
                )
            if not isinstance(doc.content, str):
                raise TypeError("PythonCodeSplitter only works with text documents (str content).")

        final_docs: list[Document] = []
        for doc in documents:
            assert doc.content is not None  # narrowed by the loop above
            if not doc.content.strip():
                logger.warning("Document ID {doc_id} has empty content. Skipping this document.", doc_id=doc.id)
                continue

            units = self._extract_units(doc.content)
            if not units:
                continue

            chunks = self._merge_units(units)
            split_id = 0
            for chunk in chunks:
                if len(chunk) == 1 and self._is_oversized(chunk[0]):
                    for piece in self._secondary_split(chunk[0], doc):
                        piece.meta["split_id"] = split_id
                        split_id += 1
                        final_docs.append(piece)
                    continue

                content = self._render_chunk_content(chunk)
                meta = self._build_chunk_meta(chunk, doc)
                meta["split_id"] = split_id
                split_id += 1
                final_docs.append(Document(content=content, meta=meta))

        return {"documents": final_docs}
