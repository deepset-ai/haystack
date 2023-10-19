# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from sample_components import StringJoiner, StringListJoiner, Hello, TextSplitter

import logging

logging.basicConfig(level=logging.DEBUG)


def test_joiner(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("hello_one", Hello())
    pipeline.add_component("hello_two", Hello())
    pipeline.add_component("hello_three", Hello())
    pipeline.add_component("joiner", StringJoiner())

    pipeline.connect("hello_one", "hello_two")
    pipeline.connect("hello_two", "joiner")
    pipeline.connect("hello_three", "joiner")

    pipeline.draw(tmp_path / "joiner_pipeline.png")

    results = pipeline.run({"hello_one": {"word": "world"}, "hello_three": {"word": "my friend"}})
    assert results == {"joiner": {"output": "Hello, my friend! Hello, Hello, world!!"}}


def test_joiner_with_lists(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("first", TextSplitter())
    pipeline.add_component("second", TextSplitter())
    pipeline.add_component("joiner", StringListJoiner())

    pipeline.connect("first", "joiner")
    pipeline.connect("second", "joiner")

    pipeline.draw(tmp_path / "joiner_list_pipeline.png")

    results = pipeline.run({"first": {"sentence": "Hello world!"}, "second": {"sentence": "How are you?"}})
    assert results == {"joiner": {"output": ["Hello", "world!", "How", "are", "you?"]}}


def test_joiner_with_pipeline_run(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("joiner", StringJoiner())
    pipeline.connect("hello", "joiner")

    pipeline.draw(tmp_path / "joiner_with_pipeline_run.png")

    results = pipeline.run({"hello": {"word": "world"}, "joiner": {"input_str": "another string!"}})
    assert results == {"joiner": {"output": "another string! Hello, world!"}}


if __name__ == "__main__":
    test_joiner(Path(__file__).parent)
    test_joiner_with_lists(Path(__file__).parent)
    test_joiner_with_pipeline_run(Path(__file__).parent)
