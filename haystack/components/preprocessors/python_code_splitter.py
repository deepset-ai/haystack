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
    """
    Internal representation of a single syntactic split unit extracted from a Python source file.

    :ivar source: The slice of the original Python source code for this unit.
    :ivar start_line: The starting line number (1-indexed) of this unit in the original source.
    :ivar end_line: The ending line number (1-indexed, inclusive) of this unit in the original source.
    :ivar kind: The kind of unit. One of: 'module_docstring', 'imports', 'class', 'class_header',
        'method', 'nested_class', 'function', or 'statement'.
    :ivar name: The name of the unit (for functions, methods, classes), or None.
    :ivar class_name: The name of the enclosing class if this unit belongs to a class, or None.
        Used only for internal bookkeeping.
    :ivar class_signature: The source text of the enclosing class signature (decorators plus the
        ``class Foo(...):`` lines, without the body), or None. Used internally to optionally
        prepend a class definition to chunks whose class header lives in another chunk.
    :ivar decorators: List of decorator strings for this unit.
    :ivar docstring: The docstring of this unit if stripped, or None.
    """

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
    Split Python source code.

    ### General behavior

    Functions of reasonable size will fit as a whole into the documents, which means,
    each created document will contain complete functions. However,
    if the number of effective lines in a function exceeds ``max_effective_lines * oversized_factor``
    (which happens for **huge functions**), the splitter falls back to the secondary
    split to split that specific function, under which syntactic correctness is not
    guaranteed anymore.

    ### Detailed description

    The component parses the source with :mod:`ast` and produces a flat list of *units*:

    - the module docstring (if present),
    - blocks of consecutive ``import`` / ``from ... import`` statements,
    - top-level functions,
    - class headers (the ``class`` signature plus any class-level statements/docstrings
      that appear before the first method),
    - class methods,
    - and any remaining module-level statements.

    The resulting chunks read top-to-bottom exactly like the original file,
    with comments and blank lines preserved.

    #### Effective lines

    Units are merged greedily in source order to reach roughly ``max_effective_lines`` per chunk
    while guaranteeing that methods and functions are fit as whole into the document.
    Sizing is measured in *effective lines*, defined as:\n
    ``ceil(len(source) / expected_chars_per_line)`` \n

    A function whose effective length exceeds ``oversized_factor * max_effective_lines`` is the only
    case where chunks may overlap: it is broken down with
    a secondary line-based split (using :class:`DocumentSplitter`) and the resulting
    chunks all carry the originating function's metadata. For the primary split, overlap
    is intentionally disabled because duplicating whole functions across chunks would be
    wasteful and produce confusing retrieval results.

    The secondary split always uses ``split_by="line"`` because other split modes (word,
    sentence, etc.) are not meaningful for code. The chunk size for the secondary split
    is controlled by ``secondary_split_length`` (defaults to ``max_effective_lines``
    when ``None``).

    #### Metadata

    Per-chunk metadata embedded on each output ``Document`` includes:

    - ``file_name`` (propagated from the input document's meta, if present),
    - ``include_classes`` (a deduplicated, source-order list of class names that the
      units in the chunk belong to, when at least one such class is involved),
    - ``decorators`` (a deduplicated, ordered list of decorator strings for the units
      in the chunk),
    - ``start_line`` and ``end_line`` in the original file (1-indexed, inclusive),
    - ``docstrings`` when ``strip_docstrings=True`` (a list of stripped docstrings in
      source order - otherwise docstrings remain in the chunk content),
    - ``unit_kinds`` (the kinds of units merged into the chunk),
    - ``source_id`` and ``split_id``.

    ### Usage example

    #### Basic use-case

    ```python
    source = '''
    \"\"\"Example module.\"\"\"
    from math import sqrt


    class Circle:
        \"\"\"A geometric circle.\"\"\"

        def __init__(self, r: float) -> None:
            self.r = r

        def area(self) -> float:
            return 3.14159 * self.r * self.r

        def circumference(self) -> float:
            return 2 * 3.14159 * self.r
    '''

    splitter = PythonCodeSplitter(min_effective_lines=4, max_effective_lines=6, preserve_class_definition=True)
    result = splitter.run(documents=[Document(content=source, meta={"file_name": "circle.py"})])
    for i, chunk in enumerate(result["documents"]):
        print(f"Chunk {i}: ")
        info = (f"Start line: {chunk.meta['start_line']}, End line: {chunk.meta['end_line']}, "
                f"include classes: {chunk.meta.get('include_classes')}")
        print(info)
        print(f"Content:\n{chunk.content}\n")
    ```

    This outputs:

    ```
    Chunk 0:
    Start line: 2, End line: 10, include classes: ['Circle']
    Content:
    \"\"\"Example module.\"\"\"
    from math import sqrt


    class Circle:
        \"\"\"A geometric circle.\"\"\"

        def __init__(self, r: float) -> None:
            self.r = r

    Chunk 1:
    Start line: 12, End line: 21, include classes: ['Circle']
    Content:
    class Circle:

        def area(self) -> float:
            return 3.14159 * self.r * self.r

        def circumference(self) -> float:
            return 2 * 3.14159 * self.r
    ```

    #### Stripping docstrings into meta

    ```python
    from haystack.components.preprocessors import PythonCodeSplitter
    from haystack import Document
    code = '''
    def add(a: float, b: float) -> float:
        \"\"\"
        Add two numbers together.

        :param a: The first number.
        :param b: The second number.
        :return: The sum of a and b.
        \"\"\"
        return a + b
    '''

    splitter = PythonCodeSplitter(strip_docstrings=True)
    result = splitter.run(documents=[Document(content=code)])
    document = result["documents"][0]
    print(f"Content:\n{document.content}")
    print("Docstring: ", document.meta.get("docstrings"))
    ```

    This outputs:

    ```
    Content:
    def add(a: float, b: float) -> float:
        return a + b

    Docstring:  ['Add two numbers together.\n\n:param a: The first number.\n:param b: ...']
    ```

    This might be useful for RAG, especially if the docstring is too large. One can strip the
    docstring to save storage, and upon embedding, one can pass `doc.meta.get("docstrings", [])`
    into `meta_fields_to_embed` field, which results in no performance degradation for
    retrieval, for example:

    ```python
    from haystack.components.preprocessors import PythonCodeSplitter
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    code = '''
    def add(a: float, b: float) -> float:
        \"\"\"
        Add two numbers together.

        :param a: The first number.
        :param b: The second number.
        :return: The sum of a and b.
        \"\"\"
        return a + b
    '''

    splitter = PythonCodeSplitter(strip_docstrings=True)
    result = splitter.run(documents=[Document(content=code)])
    document = result["documents"][0]
    print(f"Content:\n{document.content}")
    embedder = SentenceTransformersDocumentEmbedder(meta_fields_to_embed=["title"])
    docs_w_embeddings = embedder.run(documents=[document])["documents"]
    ```
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

        :param min_effective_lines: Minimum number of *effective lines* a chunk should contain.
            While the running chunk is below this threshold the splitter keeps merging
            in the next unit even if doing so moves the total away from ``max_effective_lines``.
        :param max_effective_lines: Target / maximum number of *effective lines* per chunk. Units
            are merged greedily as long as adding the next unit brings the running total
            closer to ``max_effective_lines``; once adding the next unit would move further away,
            the current chunk is flushed.
        :param expected_chars_per_line: The expected average number of characters per line, used
            to convert raw character counts into *effective lines*
            (``effective_lines = ceil(len(source) / expected_chars_per_line)``). Lines that are
            longer than this value count as more than one effective line.
        :param oversized_factor: A single function (or method) whose effective length
            exceeds ``oversized_factor * max_effective_lines`` triggers the secondary line-based
            split with overlap.
        :param strip_docstrings: If ``True``, docstrings of functions, methods and
            classes are removed from the chunk content and stored under the
            ``docstrings`` key in the chunk's meta (as a list in source order). The
            module-level docstring is left in place because it is itself a top-level
            split unit.
        :param preserve_class_definition: If ``True`` (default), every chunk that contains
            class members (methods or nested classes) but whose corresponding class header
            was emitted into a previous chunk is prefixed with the bare class signature
            (decorators plus the ``class Foo(...):`` lines) so that the enclosing class
            context is visible in the chunk. Class signatures are prepended in the source
            order in which their members first appear in the chunk; chunks that already
            contain the class header are left unchanged.
        :param secondary_split_overlap: Number of lines of overlap to use when the
            secondary (line-based) splitter is applied to oversized functions. Only
            used in the oversized fallback — the primary AST-based split never adds
            overlap.
        :param secondary_split_length: Number of lines per chunk used by the secondary
            (line-based) splitter when a function is oversized. Defaults to
            ``max_effective_lines`` when ``None``.
        :raises ValueError: If any parameter is invalid (negative, zero where positive
            is required, or ``min_effective_lines > max_effective_lines``).
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
        """
        Strip ``node``'s docstring from ``source_lines[unit_start..unit_end]`` if present.

        :returns: A tuple ``(new_source, docstring_or_None)``. If the node has no docstring
            or its docstring shares a physical line with the ``def``/``class`` statement
            (in which case removing it would break syntax), the original slice is returned
            unchanged with ``docstring_or_None=None``.
        """
        docstring = ast.get_docstring(node)
        body = node.body
        if not docstring or not body:
            return self._slice_lines(source_lines, unit_start, unit_end), None

        first = body[0]
        if not (
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        ):
            return self._slice_lines(source_lines, unit_start, unit_end), None

        # Refuse to strip if the docstring starts on the same line as the def/class
        # statement (removing it would leave broken syntax) or extends past the
        # slice that the caller asked for (e.g. a class_header slice that ends
        # before the docstring's last line).
        ds_start = first.lineno
        ds_end = first.end_lineno or first.lineno
        if ds_start <= node.lineno or ds_end > unit_end:
            return self._slice_lines(source_lines, unit_start, unit_end), None

        before = source_lines[unit_start - 1 : ds_start - 1]
        after = source_lines[ds_end:unit_end]
        return "".join(before + after), docstring

    def _emit_class_units(self, cls: ast.ClassDef, source_lines: list[str], cursor: int, units: list[_CodeUnit]) -> int:
        """
        Emit the class header and per-method units for ``cls``.

        :returns: The new cursor (1-indexed, next line not yet emitted) after the class.
        """
        class_start = cls.decorator_list[0].lineno if cls.decorator_list else cls.lineno
        class_end = cls.end_lineno or cls.lineno
        class_name = cls.name
        class_decorators = [self._safe_unparse(d) for d in cls.decorator_list]

        # Identify body children that should become their own units. Functions, async
        # functions, and nested classes are split out so that "a method should not be
        # broken mid-statement" holds at this level too.
        split_children_idx = [
            k
            for k, child in enumerate(cls.body)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        # Bare class signature lines (decorators + ``class Foo(...):``), i.e. everything
        # from the start of the class up to the line where the body begins. Used by
        # ``preserve_class_definition`` to prepend the enclosing class context to chunks
        # whose class header lives in a previous chunk.
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
                continue  # narrowed by split_children_idx; kept for the type checker
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

        # Sweep up any trailing class-body lines (comments, blank lines, dangling
        # class-level statements after the last method) so we don't lose them.
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

            # Consecutive imports are grouped into a single unit — they are tightly
            # related context and chopping them up is rarely useful.
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

        # Append trailing content (e.g. a trailing comment after the last node) to the
        # last unit so the splitter is loss-less w.r.t. the original source.
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
        """
        Greedily merge ``units`` (in source order) into chunks of roughly ``max_effective_lines``.

        A unit whose effective length exceeds ``oversized_factor * max_effective_lines`` is
        emitted as its own (single-unit) chunk so that the caller can apply the
        secondary line-based split with overlap; it is never merged with neighbours.
        """
        chunks: list[list[_CodeUnit]] = []
        current: list[_CodeUnit] = []
        current_lines = 0
        target = self.max_effective_lines

        def flush() -> None:
            """Flush the current chunk (if any) into ``chunks`` and reset the running state."""
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

            new_total = current_lines + unit_eff
            # Keep merging while we are below the minimum, or while adding the next
            # unit brings the running total closer to the target than leaving it out.
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
        """
        Build the textual content for ``chunk``.

        When ``preserve_class_definition`` is enabled, every class whose members
        (methods or nested classes) appear in the chunk but whose ``class`` /
        ``class_header`` unit does not, gets its bare signature prepended so that
        the enclosing class context remains visible in the chunk.
        """
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

        # split_length is interpreted in physical lines by DocumentSplitter; this is an
        # approximation of `max_effective_lines` effective lines but is good enough for the
        # fallback path (and the warning above flags that the result is approximate).
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
        Split each Python source ``Document`` into Python syntax-aware chunks.

        :param documents: Documents whose ``content`` is Python source code. Each
            document's ``meta`` is propagated onto its chunks; if ``file_name`` is
            present in the meta it is preserved on every output chunk.
        :returns: A dictionary of the form ``{"documents": [...]}``, where each chunk's
            meta additionally carries ``source_id``, ``split_id``, ``start_line``,
            ``end_line``, ``unit_kinds``
            and - where applicable - ``include_classes``, ``decorators``, ``docstrings``,
            ``secondary_split``.
        :raises ValueError: If any document's content is ``None``.
        :raises TypeError: If any document's content is not a string.
        :raises SyntaxError: If a document's content is not valid Python (raised by
            :func:`ast.parse`).
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
