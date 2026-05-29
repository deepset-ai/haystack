# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import textwrap

import pytest

from haystack import Document
from haystack.components.preprocessors import PythonCodeSplitter


@pytest.fixture
def simple_module_source():
    return textwrap.dedent(
        '''
        """Example module docstring."""
        import os
        import sys
        from math import sqrt


        def add(a, b):
            """Add two numbers."""
            return a + b


        def subtract(a, b):
            """Subtract two numbers."""
            return a - b
        '''
    ).lstrip()


@pytest.fixture
def class_source():
    return textwrap.dedent(
        '''
        """Geometry helpers."""
        from math import pi


        class Shape:
            """Base shape."""

            kind = "shape"

            def __init__(self, name: str) -> None:
                self.name = name

            def describe(self) -> str:
                return f"shape {self.name}"


        class Circle(Shape, metaclass=type):
            """A circle."""

            def __init__(self, r: float) -> None:
                super().__init__("circle")
                self.r = r

            @staticmethod
            def pi_value() -> float:
                return pi

            @classmethod
            def unit(cls) -> "Circle":
                return cls(1.0)

            def area(self) -> float:
                return pi * self.r * self.r
        '''
    ).lstrip()


@pytest.fixture
def oversized_function_source():
    # A function whose body has many lines so that
    # effective_lines >> oversized_factor * max_effective_lines.
    body_lines = "\n".join(f"    x_{i} = {i}" for i in range(200))
    return f"def giant():\n{body_lines}\n    return x_0\n"


class TestInitValidation:
    def test_defaults(self):
        splitter = PythonCodeSplitter()
        assert splitter.min_effective_lines == 20
        assert splitter.max_effective_lines == 100
        assert splitter.expected_chars_per_line == 45
        assert splitter.oversized_factor == 3
        assert splitter.strip_docstrings is False
        assert splitter.preserve_class_definition is True
        assert splitter.secondary_split_overlap == 5
        assert splitter.secondary_split_length is None

    def test_custom_values(self):
        splitter = PythonCodeSplitter(
            min_effective_lines=2,
            max_effective_lines=10,
            expected_chars_per_line=80,
            oversized_factor=4,
            strip_docstrings=True,
            preserve_class_definition=False,
            secondary_split_overlap=2,
            secondary_split_length=15,
        )
        assert splitter.min_effective_lines == 2
        assert splitter.max_effective_lines == 10
        assert splitter.expected_chars_per_line == 80
        assert splitter.oversized_factor == 4
        assert splitter.strip_docstrings is True
        assert splitter.preserve_class_definition is False
        assert splitter.secondary_split_overlap == 2
        assert splitter.secondary_split_length == 15

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_effective_lines": 0},
            {"min_effective_lines": -1},
            {"max_effective_lines": 0},
            {"max_effective_lines": -3},
            {"min_effective_lines": 10, "max_effective_lines": 5},
            {"expected_chars_per_line": 0},
            {"oversized_factor": 0},
            {"secondary_split_overlap": -1},
            {"secondary_split_length": -1},
        ],
    )
    def test_invalid_init_raises(self, kwargs):
        with pytest.raises(ValueError):
            PythonCodeSplitter(**kwargs)


class TestRunInputValidation:
    def test_none_content_raises_value_error(self):
        splitter = PythonCodeSplitter()
        doc = Document(content=None)
        with pytest.raises(ValueError):
            splitter.run(documents=[doc])

    def test_non_string_content_raises_type_error(self):
        splitter = PythonCodeSplitter()
        # Document normally coerces, so bypass via construction-friendly path.
        doc = Document(content="placeholder")
        doc.content = 12345  # type: ignore[assignment]
        with pytest.raises(TypeError):
            splitter.run(documents=[doc])

    def test_invalid_syntax_raises(self):
        splitter = PythonCodeSplitter()
        doc = Document(content="def broken(:\n    pass\n")
        with pytest.raises(SyntaxError):
            splitter.run(documents=[doc])

    def test_empty_documents_list(self):
        splitter = PythonCodeSplitter()
        result = splitter.run(documents=[])
        assert result == {"documents": []}


class TestBasicOutput:
    def test_returns_dict_with_documents(self, simple_module_source):
        splitter = PythonCodeSplitter()
        result = splitter.run(documents=[Document(content=simple_module_source)])
        assert isinstance(result, dict)
        assert "documents" in result
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) >= 1
        for chunk in result["documents"]:
            assert isinstance(chunk, Document)
            assert isinstance(chunk.content, str)
            assert chunk.content  # checks non-empty

    def test_split_id_starts_at_zero_and_increments(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=class_source)])
        chunks = result["documents"]
        ids = [c.meta["split_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_source_id_consistent_within_one_document(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=class_source)])
        chunks = result["documents"]
        source_ids = {c.meta["source_id"] for c in chunks}
        assert len(source_ids) == 1

    def test_source_id_differs_between_documents(self, simple_module_source, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=10)
        docs = [Document(content=simple_module_source), Document(content=class_source)]
        result = splitter.run(documents=docs)
        source_ids = {c.meta["source_id"] for c in result["documents"]}
        assert len(source_ids) == 2

    def test_chunks_have_required_meta_fields(self, simple_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=10)
        result = splitter.run(documents=[Document(content=simple_module_source)])
        for chunk in result["documents"]:
            assert "source_id" in chunk.meta
            assert "split_id" in chunk.meta
            assert "start_line" in chunk.meta
            assert "end_line" in chunk.meta
            assert "unit_kinds" in chunk.meta
            assert isinstance(chunk.meta["unit_kinds"], list)
            assert chunk.meta["start_line"] >= 1
            assert chunk.meta["end_line"] >= chunk.meta["start_line"]

    def test_unit_kinds_lists_what_was_merged(self, simple_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=200)
        result = splitter.run(documents=[Document(content=simple_module_source)])
        all_kinds = set()
        for chunk in result["documents"]:
            for kind in chunk.meta["unit_kinds"]:
                all_kinds.add(kind)
        text = " ".join(all_kinds).lower()
        assert any("import" in t for t in all_kinds) or "import" in text
        assert any("func" in t or "method" in t for t in all_kinds) or any("func" in t for t in text.split())

    def test_multiple_documents_each_produces_chunks(self, simple_module_source, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=10)
        docs = [
            Document(content=simple_module_source, meta={"file_name": "a.py"}),
            Document(content=class_source, meta={"file_name": "b.py"}),
        ]
        result = splitter.run(documents=docs)
        file_names = {c.meta.get("file_name") for c in result["documents"]}
        assert file_names == {"a.py", "b.py"}

    def test_split_id_resets_per_document(self, simple_module_source, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=10)
        docs = [
            Document(content=simple_module_source, meta={"file_name": "a.py"}),
            Document(content=class_source, meta={"file_name": "b.py"}),
        ]
        result = splitter.run(documents=docs)
        per_file_ids: dict[str, list[int]] = {"a.py": [], "b.py": []}
        for chunk in result["documents"]:
            per_file_ids[chunk.meta["file_name"]].append(chunk.meta["split_id"])
        for ids in per_file_ids.values():
            assert ids == sorted(ids)
            assert ids[0] == 0


class TestOrderingAndLineRanges:
    def test_chunks_are_in_source_order(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=class_source)])
        chunks = result["documents"]
        # start_line should be non-decreasing across chunks
        prev_start = 0
        for chunk in chunks:
            assert chunk.meta["start_line"] >= prev_start
            assert chunk.meta["end_line"] >= chunk.meta["start_line"]
            prev_start = chunk.meta["start_line"]

    def test_chunks_dont_overlap_in_primary_split(self, class_source):
        # Primary AST split has no overlap; adjacent chunks should be disjoint.
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=class_source)])
        chunks = result["documents"]
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            assert nxt.meta["start_line"] > prev.meta["end_line"]

    def test_chunks_read_top_to_bottom(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=class_source)])
        source_lines = class_source.splitlines(keepends=True)
        for chunk in result["documents"]:
            assert chunk.content is not None
            start = chunk.meta["start_line"]
            end = chunk.meta["end_line"]
            expected_slice = "".join(source_lines[start - 1 : end])
            # The exact source slice for [start_line, end_line] must appear contiguously
            # in the chunk content. With preserve_class_definition=True (the default) the
            # chunk may additionally carry a prepended class signature, but the slice
            # itself must be present byte-for-byte - a regression that dropped or
            # reordered lines within the range would fail this assertion.
            assert expected_slice in chunk.content, (
                f"Chunk content is missing the source slice for lines {start}-{end}.\n"
                f"--- expected slice ---\n{expected_slice!r}\n"
                f"--- chunk content ---\n{chunk.content!r}"
            )

    def test_chunks_equal_source_slice_without_class_preservation(self, class_source):
        # With preserve_class_definition=False there is no prepended signature, so
        # the chunk content for [start_line, end_line] must equal the source slice
        # minus at most a fixed-length preamble that bridges from the previous unit's
        # end. We assert that the chunk content ends exactly with the slice and that
        # nothing inside it is rewritten.
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5, preserve_class_definition=False)
        result = splitter.run(documents=[Document(content=class_source)])
        source_lines = class_source.splitlines(keepends=True)
        for chunk in result["documents"]:
            assert chunk.content is not None
            start = chunk.meta["start_line"]
            end = chunk.meta["end_line"]
            expected_slice = "".join(source_lines[start - 1 : end])
            assert chunk.content.endswith(expected_slice), (
                f"Chunk content does not end with the source slice for lines {start}-{end}.\n"
                f"--- expected slice ---\n{expected_slice!r}\n"
                f"--- chunk content ---\n{chunk.content!r}"
            )


class TestFileNamePropagation:
    def test_file_name_propagated_to_all_chunks(self, simple_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=simple_module_source, meta={"file_name": "sample.py"})])
        for chunk in result["documents"]:
            assert chunk.meta["file_name"] == "sample.py"

    def test_no_file_name_when_absent(self, simple_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=simple_module_source)])
        for chunk in result["documents"]:
            assert "file_name" not in chunk.meta

    def test_other_meta_is_propagated(self, simple_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5)
        result = splitter.run(
            documents=[Document(content=simple_module_source, meta={"file_name": "x.py", "project": "haystack"})]
        )
        for chunk in result["documents"]:
            assert chunk.meta["project"] == "haystack"


class TestDecorators:
    def test_decorators_metadata_present(self):
        source = textwrap.dedent(
            """
            class A:
                @staticmethod
                def s():
                    return 1

                @classmethod
                def c(cls):
                    return 2
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=2)
        result = splitter.run(documents=[Document(content=source)])
        all_decorators = []
        for chunk in result["documents"]:
            all_decorators.extend(chunk.meta.get("decorators") or [])
        assert any("staticmethod" in d for d in all_decorators)
        assert any("classmethod" in d for d in all_decorators)

    def test_decorator_lines_included_in_chunk_content(self):
        source = textwrap.dedent(
            """
            class A:
                @staticmethod
                def s():
                    return 1
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=2)
        result = splitter.run(documents=[Document(content=source)])
        # The chunk holding `def s` must also contain the `@staticmethod` line.
        chunks_with_s = [c for c in result["documents"] if "def s" in (c.content or "")]
        assert chunks_with_s
        for chunk in chunks_with_s:
            assert "@staticmethod" in (chunk.content or "")

    def test_decorators_deduped_in_chunk(self):
        # Two methods sharing the same decorator should not list the
        # decorator twice in the chunk's meta if they end up merged.
        source = textwrap.dedent(
            """
            class A:
                @staticmethod
                def one():
                    return 1

                @staticmethod
                def two():
                    return 2
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=20)
        result = splitter.run(documents=[Document(content=source)])
        for chunk in result["documents"]:
            decorators = chunk.meta.get("decorators") or []
            assert len(decorators) == len(set(decorators))

    def test_function_with_three_decorators_lists_all(self):
        source = textwrap.dedent(
            """
            def deco_a(fn):
                return fn


            def deco_b(fn):
                return fn


            def deco_c(fn):
                return fn


            @deco_a
            @deco_b
            @deco_c
            def triple():
                return 1
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=200)
        result = splitter.run(documents=[Document(content=source)])

        chunks_with_triple = [c for c in result["documents"] if "def triple" in (c.content or "")]
        assert chunks_with_triple, "Expected the `triple` function to appear in a chunk"

        decorators = []
        for chunk in chunks_with_triple:
            decorators.extend(chunk.meta.get("decorators") or [])

        joined = " ".join(decorators)
        assert "deco_a" in joined
        assert "deco_b" in joined
        assert "deco_c" in joined

    def test_function_with_three_decorators_all_lines_in_content(self):
        source = textwrap.dedent(
            """
            @deco_a
            @deco_b
            @deco_c
            def triple():
                return 1
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=50)
        result = splitter.run(documents=[Document(content=source)])

        chunks_with_triple = [c for c in result["documents"] if "def triple" in (c.content or "")]
        assert chunks_with_triple
        for chunk in chunks_with_triple:
            content = chunk.content or ""
            assert "@deco_a" in content
            assert "@deco_b" in content
            assert "@deco_c" in content


class TestIncludeClassesMeta:
    def test_circle_methods_carry_include_classes(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3)
        result = splitter.run(documents=[Document(content=class_source)])
        circle_chunks = [c for c in result["documents"] if "Circle" in (c.meta.get("include_classes") or [])]
        assert circle_chunks, "Expected at least one chunk for class Circle"

    def test_shape_methods_carry_include_classes(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3)
        result = splitter.run(documents=[Document(content=class_source)])
        shape_chunks = [c for c in result["documents"] if "Shape" in (c.meta.get("include_classes") or [])]
        assert shape_chunks, "Expected at least one chunk for class Shape"

    def test_include_classes_set_for_class_chunks(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3)
        result = splitter.run(documents=[Document(content=class_source)])

        all_classes = set()
        for chunk in result["documents"]:
            for cls in chunk.meta.get("include_classes") or []:
                all_classes.add(cls)
        assert "Shape" in all_classes
        assert "Circle" in all_classes

    def test_include_classes_absent_when_no_class_involved(self, simple_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=200)
        result = splitter.run(documents=[Document(content=simple_module_source)])
        for chunk in result["documents"]:
            include_classes = chunk.meta.get("include_classes")
            assert not include_classes

    def test_include_classes_is_deduplicated(self):
        source = textwrap.dedent(
            """
            class A:
                def one(self):
                    return 1

                def two(self):
                    return 2

                def three(self):
                    return 3
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=200)
        result = splitter.run(documents=[Document(content=source)])
        for chunk in result["documents"]:
            include_classes = chunk.meta.get("include_classes") or []
            assert len(include_classes) == len(set(include_classes))

    def test_include_classes_preserves_source_order(self):
        source = textwrap.dedent(
            """
            class First:
                def f(self):
                    return 1


            class Second:
                def g(self):
                    return 2


            class Third:
                def h(self):
                    return 3
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=500)
        result = splitter.run(documents=[Document(content=source)])

        for chunk in result["documents"]:
            include_classes = chunk.meta.get("include_classes") or []
            if len(include_classes) >= 2:
                expected_order = [c for c in ["First", "Second", "Third"] if c in include_classes]
                assert include_classes == expected_order


class TestPreserveClassDefinition:
    @pytest.fixture
    def multi_method_class_source(self):
        return textwrap.dedent(
            '''
            class Greeter:
                """A friendly greeter."""

                kind = "greeter"

                def __init__(self, name: str) -> None:
                    self.name = name

                def hello(self) -> str:
                    return f"hello {self.name}"

                def bye(self) -> str:
                    return f"bye {self.name}"

                def shout(self) -> str:
                    return f"HELLO {self.name.upper()}"

                def whisper(self) -> str:
                    return f"hello {self.name}..."
            '''
        ).lstrip()

    def test_default_preserve_class_definition_is_true(self):
        splitter = PythonCodeSplitter()
        assert splitter.preserve_class_definition is True

    def test_class_signature_prepended_to_later_chunks(self, multi_method_class_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3, preserve_class_definition=True)
        result = splitter.run(documents=[Document(content=multi_method_class_source)])
        chunks = result["documents"]
        assert len(chunks) >= 2, "Need multiple chunks for this test to be meaningful"

        # Every chunk that includes a method of Greeter must show `class Greeter`.
        for chunk in chunks:
            content = chunk.content or ""
            include_classes = chunk.meta.get("include_classes") or []
            if "Greeter" in include_classes:
                assert "class Greeter" in content, (
                    f"Expected 'class Greeter' to be preserved in chunk content but got:\n{content}"
                )

    def test_disabled_does_not_prepend_class_signature(self, multi_method_class_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3, preserve_class_definition=False)
        result = splitter.run(documents=[Document(content=multi_method_class_source)])
        chunks = result["documents"]
        # Exactly one chunk should contain the original `class Greeter` header
        # (the one that actually produced from the class header unit).
        chunks_with_header = [c for c in chunks if "class Greeter" in (c.content or "")]
        assert len(chunks_with_header) == 1, (
            "With preserve_class_definition=False, only the original chunk should contain the class header."
        )

    def test_preserve_does_not_duplicate_header_in_original_chunk(self, multi_method_class_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3, preserve_class_definition=True)
        result = splitter.run(documents=[Document(content=multi_method_class_source)])
        for chunk in result["documents"]:
            content = chunk.content or ""
            # The class header should never appear more than once in a single chunk.
            assert content.count("class Greeter") <= 1

    def test_preserve_keeps_inheritance_and_metaclass(self):
        source = textwrap.dedent(
            '''
            class Base:
                pass


            class Meta(type):
                pass


            class Child(Base, metaclass=Meta):
                """A child class."""

                def one(self):
                    return 1

                def two(self):
                    return 2

                def three(self):
                    return 3

                def four(self):
                    return 4
            '''
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3, preserve_class_definition=True)
        result = splitter.run(documents=[Document(content=source)])

        child_chunks = [c for c in result["documents"] if "Child" in (c.meta.get("include_classes") or [])]
        assert len(child_chunks) >= 2, "Need multiple Child chunks for this test"

        # Every chunk that contains a member of Child must show the full class
        # signature, including base classes and metaclass.
        for chunk in child_chunks:
            content = chunk.content or ""
            assert "class Child(Base, metaclass=Meta):" in content, (
                f"Expected the full class signature (with bases and metaclass) to be preserved, got:\n{content}"
            )

    def test_preserve_keeps_decorators_on_class(self):
        source = textwrap.dedent(
            """
            def reg(cls):
                return cls


            @reg
            class Decorated:
                def one(self):
                    return 1

                def two(self):
                    return 2

                def three(self):
                    return 3

                def four(self):
                    return 4
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3, preserve_class_definition=True)
        result = splitter.run(documents=[Document(content=source)])

        decorated_chunks = [c for c in result["documents"] if "Decorated" in (c.meta.get("include_classes") or [])]
        assert len(decorated_chunks) >= 2

        for chunk in decorated_chunks:
            content = chunk.content or ""
            assert "class Decorated" in content
            assert "@reg" in content, f"Expected '@reg' decorator to be preserved on class signature, got:\n{content}"

    def test_preserve_handles_multiple_classes(self):
        source = textwrap.dedent(
            """
            class A:
                def a1(self):
                    return 1

                def a2(self):
                    return 2

                def a3(self):
                    return 3


            class B:
                def b1(self):
                    return 1

                def b2(self):
                    return 2

                def b3(self):
                    return 3
            """
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=3, preserve_class_definition=True)
        result = splitter.run(documents=[Document(content=source)])

        for chunk in result["documents"]:
            content = chunk.content or ""
            include_classes = chunk.meta.get("include_classes") or []
            for cls in include_classes:
                assert f"class {cls}" in content, (
                    f"Expected 'class {cls}' to be preserved in chunk content but got:\n{content}"
                )


class TestDocstringStripping:
    def test_default_keeps_docstrings_in_content(self, class_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=8)
        result = splitter.run(documents=[Document(content=class_source)])
        contents = "\n".join(c.content or "" for c in result["documents"])
        assert "A circle." in contents
        # When not stripping, meta should not carry a docstrings list.
        for chunk in result["documents"]:
            assert "docstrings" not in chunk.meta or not chunk.meta.get("docstrings")

    def test_strip_docstrings_moves_them_to_meta(self):
        source = textwrap.dedent(
            '''
            def foo():
                """Foo docstring."""
                return 1


            def bar():
                """Bar docstring."""
                return 2
            '''
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=10, strip_docstrings=True)
        result = splitter.run(documents=[Document(content=source)])
        # Collect docstrings from meta
        all_docstrings = []
        for chunk in result["documents"]:
            all_docstrings.extend(chunk.meta.get("docstrings") or [])
        joined_docstrings = " | ".join(all_docstrings)
        assert "Foo docstring." in joined_docstrings
        assert "Bar docstring." in joined_docstrings
        # And the docstring text should NOT remain in any chunk content
        joined_content = "\n".join(c.content or "" for c in result["documents"])
        assert "Foo docstring." not in joined_content
        assert "Bar docstring." not in joined_content

    def test_strip_docstrings_preserves_module_docstring(self):
        source = textwrap.dedent(
            '''
            """Module-level docstring."""

            def foo():
                """Inner."""
                return 1
            '''
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=20, strip_docstrings=True)
        result = splitter.run(documents=[Document(content=source)])
        joined = "\n".join(c.content or "" for c in result["documents"])
        # Module docstring should still appear (it's itself a unit).
        assert "Module-level docstring." in joined

    def test_strip_class_header_docstring_moves_to_meta(self):
        source = textwrap.dedent(
            '''
            class MyClass:
                """Class-level docstring."""

                class_var = 42

                def method(self):
                    return self.class_var
            '''
        ).lstrip()
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=10, strip_docstrings=True)
        result = splitter.run(documents=[Document(content=source)])

        header_chunks = [c for c in result["documents"] if "class_header" in c.meta.get("unit_kinds", [])]
        assert header_chunks, "expected at least one class_header chunk"

        header = header_chunks[0]
        # Docstring text must not appear in the chunk content.
        assert "Class-level docstring." not in (header.content or "")
        # Docstring must be captured in meta instead.
        assert "Class-level docstring." in " | ".join(header.meta.get("docstrings") or [])


class TestTopLevelStatements:
    @pytest.fixture
    def rich_module_source(self):
        return textwrap.dedent(
            '''
            """Utility helpers for the pipeline."""
            import os
            import sys
            from pathlib import Path

            MAX_RETRIES = 3
            DEFAULT_TIMEOUT = 30.0
            LOG_PREFIX = "app"


            def process(data):
                """Process data."""
                result = data.strip()
                return result


            def validate(value):
                """Validate a value."""
                if value is None:
                    raise ValueError("value cannot be None")
                return True


            class Manager:
                """Resource manager."""

                def __init__(self):
                    self.items = []

                def add(self, item):
                    self.items.append(item)

                def remove(self, item):
                    self.items.remove(item)


            if __name__ == "__main__":
                mgr = Manager()
                mgr.add(process("test"))
            '''
        ).lstrip()

    def test_statement_unit_kind_present(self, rich_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=50)
        result = splitter.run(documents=[Document(content=rich_module_source)])
        all_kinds = [k for c in result["documents"] for k in c.meta.get("unit_kinds", [])]
        assert "statement" in all_kinds

    def test_import_unit_kind_present(self, rich_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=50)
        result = splitter.run(documents=[Document(content=rich_module_source)])
        all_kinds = [k for c in result["documents"] for k in c.meta.get("unit_kinds", [])]
        assert "imports" in all_kinds

    def test_module_docstring_unit_kind_present(self, rich_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=50)
        result = splitter.run(documents=[Document(content=rich_module_source)])
        all_kinds = [k for c in result["documents"] for k in c.meta.get("unit_kinds", [])]
        assert "module_docstring" in all_kinds

    def test_first_chunk_contains_preamble_statements(self, rich_module_source):
        # The preamble (module docstring + 3 imports + 3 assignments) totals 6 effective
        # lines at the default expected_chars_per_line=45.  Setting max_effective_lines=6
        # causes the greedy merger to flush before absorbing the first function definition,
        # so the first chunk must be exactly the pre-function preamble.
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=6)
        result = splitter.run(documents=[Document(content=rich_module_source)])
        assert len(result["documents"]) >= 2, "Need at least two chunks for this assertion"
        first = result["documents"][0]
        content = first.content or ""
        assert '"""Utility helpers for the pipeline."""' in content
        assert "import os" in content
        assert "import sys" in content
        assert "from pathlib import Path" in content
        assert "MAX_RETRIES = 3" in content
        assert "DEFAULT_TIMEOUT = 30.0" in content
        assert 'LOG_PREFIX = "app"' in content
        assert "def process" not in content
        assert "def validate" not in content
        assert "class Manager" not in content

    def test_if_main_produces_statement_unit(self, rich_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=50)
        result = splitter.run(documents=[Document(content=rich_module_source)])
        all_kinds = [k for c in result["documents"] for k in c.meta.get("unit_kinds", [])]
        assert "statement" in all_kinds
        joined = "\n".join(c.content or "" for c in result["documents"])
        assert 'if __name__ == "__main__"' in joined

    def test_all_imports_appear_in_output(self, rich_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=50)
        result = splitter.run(documents=[Document(content=rich_module_source)])
        joined = "\n".join(c.content or "" for c in result["documents"])
        assert "import os" in joined
        assert "import sys" in joined
        assert "from pathlib import Path" in joined

    def test_module_docstring_text_preserved(self, rich_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=50)
        result = splitter.run(documents=[Document(content=rich_module_source)])
        joined = "\n".join(c.content or "" for c in result["documents"])
        assert "Utility helpers for the pipeline." in joined


class TestOversizedFallback:
    def test_warns_on_oversized_function(self, oversized_function_source, caplog):
        import logging

        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5, oversized_factor=3)

        with caplog.at_level(logging.WARNING):
            result = splitter.run(documents=[Document(content=oversized_function_source)])

        text = caplog.text.lower()
        assert "oversiz" in text or "secondary" in text, "Splitter should warn about oversized function fallback"
        assert len(result["documents"]) >= 2

    def test_oversized_chunks_marked_with_secondary_split(self, oversized_function_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5, oversized_factor=3)
        result = splitter.run(documents=[Document(content=oversized_function_source)])
        secondary = [c for c in result["documents"] if c.meta.get("secondary_split")]
        assert secondary, "Oversized function chunks must carry secondary_split metadata"

    def test_oversized_chunks_belong_to_same_function(self, oversized_function_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=5, oversized_factor=3)
        result = splitter.run(documents=[Document(content=oversized_function_source)])
        secondary = [c for c in result["documents"] if c.meta.get("secondary_split")]
        # All secondary chunks should fall within the original function's line range.
        for chunk in secondary:
            assert chunk.meta["start_line"] >= 1
            assert chunk.meta["end_line"] <= oversized_function_source.count("\n") + 1

    def test_small_function_does_not_trigger_secondary(self, simple_module_source):
        splitter = PythonCodeSplitter(min_effective_lines=2, max_effective_lines=50, oversized_factor=3)
        result = splitter.run(documents=[Document(content=simple_module_source)])
        for chunk in result["documents"]:
            assert not chunk.meta.get("secondary_split")

    def test_long_lines_count_as_more_effective_lines(self):
        long_line_source = (
            textwrap.dedent(
                """
            def short():
                return 1


            def longer():
                return "{padding}"
            """
            )
            .lstrip()
            .format(padding="x" * 500)
        )
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=2, expected_chars_per_line=10)
        result = splitter.run(documents=[Document(content=long_line_source)])
        chunks_with_short = [c for c in result["documents"] if "def short" in (c.content or "")]
        chunks_with_long = [c for c in result["documents"] if "def longer" in (c.content or "")]
        assert chunks_with_short and chunks_with_long
        assert chunks_with_short[0] is not chunks_with_long[0]


class TestEdgeCases:
    def test_module_with_only_docstring(self):
        source = '"""Just a docstring."""\n'
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=source)])
        assert len(result["documents"]) == 1
        assert "Just a docstring." in (result["documents"][0].content or "")

    def test_module_with_only_imports(self):
        source = "import os\nimport sys\nfrom math import pi\n"
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=source)])
        joined = "\n".join(c.content or "" for c in result["documents"])
        assert "import os" in joined
        assert "import sys" in joined
        assert "from math import pi" in joined

    def test_module_with_only_one_function(self):
        source = "def f():\n    return 1\n"
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=5)
        result = splitter.run(documents=[Document(content=source)])
        assert len(result["documents"]) == 1
        assert "def f" in (result["documents"][0].content or "")

    def test_empty_string_source(self):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=5)
        # Empty source is valid Python (parses to an empty module).
        result = splitter.run(documents=[Document(content="")])
        # Either yields no chunks or a single empty-ish chunk; both are acceptable
        # but the call must not raise.
        assert isinstance(result["documents"], list)

    def test_invalid_syntax_raises_syntax_error(self):
        splitter = PythonCodeSplitter(min_effective_lines=1, max_effective_lines=5)
        doc = Document(content="class Broken(\n    pass\n")
        with pytest.raises(SyntaxError):
            splitter.run(documents=[doc])
