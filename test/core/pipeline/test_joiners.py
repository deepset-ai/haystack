# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import Hello, StringJoiner, StringListJoiner, TextSplitter

logging.basicConfig(level=logging.DEBUG)


def test_joiner():
    pipeline = Pipeline()
    hello_one = Hello()
    hello_two = Hello()
    hello_three = Hello()
    joiner = StringJoiner()
    pipeline.add_component("hello_one", hello_one)
    pipeline.add_component("hello_two", hello_two)
    pipeline.add_component("hello_three", hello_three)
    pipeline.add_component("joiner", joiner)

    pipeline.connect(hello_one.outputs.output, hello_two.inputs.word)
    pipeline.connect(hello_two.outputs.output, joiner.inputs.input_str)
    pipeline.connect(hello_three.outputs.output, joiner.inputs.input_str)

    results = pipeline.run({"hello_one": {"word": "world"}, "hello_three": {"word": "my friend"}})
    assert results == {"joiner": {"output": "Hello, my friend! Hello, Hello, world!!"}}


def test_joiner_with_lists():
    pipeline = Pipeline()
    first = TextSplitter()
    second = TextSplitter()
    joiner = StringListJoiner()
    pipeline.add_component("first", first)
    pipeline.add_component("second", second)
    pipeline.add_component("joiner", joiner)

    pipeline.connect(first.outputs.output, joiner.inputs.inputs)
    pipeline.connect(second.outputs.output, joiner.inputs.inputs)

    results = pipeline.run({"first": {"sentence": "Hello world!"}, "second": {"sentence": "How are you?"}})
    assert results == {"joiner": {"output": ["Hello", "world!", "How", "are", "you?"]}}


def test_joiner_with_pipeline_run():
    pipeline = Pipeline()
    hello = Hello()
    joiner = StringJoiner()
    pipeline.add_component("hello", hello)
    pipeline.add_component("joiner", joiner)
    pipeline.connect(hello.outputs.output, joiner.inputs.input_str)

    results = pipeline.run({"hello": {"word": "world"}, "joiner": {"input_str": "another string!"}})
    assert results == {"joiner": {"output": "another string! Hello, world!"}}
