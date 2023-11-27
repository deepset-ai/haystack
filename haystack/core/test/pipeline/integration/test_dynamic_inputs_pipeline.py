from pathlib import Path
from canals import Pipeline
from sample_components import FString, Hello, TextSplitter


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("fstring", FString(template="This is the greeting: {greeting}!", variables=["greeting"]))
    pipeline.add_component("splitter", TextSplitter())
    pipeline.connect("hello.output", "fstring.greeting")
    pipeline.connect("fstring.string", "splitter.sentence")

    pipeline.draw(tmp_path / "dynamic_inputs_pipeline.png")

    output = pipeline.run({"hello": {"word": "Alice"}})
    assert output == {"splitter": {"output": ["This", "is", "the", "greeting:", "Hello,", "Alice!!"]}}

    output = pipeline.run({"hello": {"word": "Alice"}, "fstring": {"template": "Received: {greeting}"}})
    assert output == {"splitter": {"output": ["Received:", "Hello,", "Alice!"]}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
