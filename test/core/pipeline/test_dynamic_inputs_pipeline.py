from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import FString, Hello, TextSplitter


def test_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("fstring", FString(template="This is the greeting: {greeting}!", variables=["greeting"]))
    pipeline.add_component("splitter", TextSplitter())
    pipeline.connect("hello.output", "fstring.greeting")
    pipeline.connect("fstring.string", "splitter.sentence")

    output = pipeline.run({"hello": {"word": "Alice"}})
    assert output == {"splitter": {"output": ["This", "is", "the", "greeting:", "Hello,", "Alice!!"]}}

    output = pipeline.run({"hello": {"word": "Alice"}, "fstring": {"template": "Received: {greeting}"}})
    assert output == {"splitter": {"output": ["Received:", "Hello,", "Alice!"]}}
