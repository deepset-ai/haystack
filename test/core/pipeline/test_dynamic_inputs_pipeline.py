from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import FString, Hello, TextSplitter


def test_pipeline():
    pipeline = Pipeline()
    hello = Hello()
    fstring = FString(template="This is the greeting: {greeting}!", variables=["greeting"])
    splitter = TextSplitter()
    pipeline.add_component("hello", hello)
    pipeline.add_component("fstring", fstring)
    pipeline.add_component("splitter", splitter)
    pipeline.connect(hello.outputs.output, fstring.inputs.greeting)
    pipeline.connect(fstring.outputs.string, splitter.inputs.sentence)

    output = pipeline.run({"hello": {"word": "Alice"}})
    assert output == {"splitter": {"output": ["This", "is", "the", "greeting:", "Hello,", "Alice!!"]}}

    output = pipeline.run({"hello": {"word": "Alice"}, "fstring": {"template": "Received: {greeting}"}})
    assert output == {"splitter": {"output": ["Received:", "Hello,", "Alice!"]}}
