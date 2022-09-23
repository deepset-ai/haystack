import os
from pathlib import Path

import pytest

from haystack.nodes import FARMReader


@pytest.mark.parametrize("model_name", ["deepset/roberta-base-squad2", "deepset/bert-base-uncased-squad2"])
def test_farm_reader_onnx_conversion_and_inference(model_name, tmpdir, docs):
    FARMReader.convert_to_onnx(model_name=model_name, output_path=Path(tmpdir, "onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "model.onnx"))
    assert os.path.exists(Path(tmpdir, "onnx", "processor_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "onnx_model_config.json"))
    assert os.path.exists(Path(tmpdir, "onnx", "language_model_config.json"))

    reader = FARMReader(str(Path(tmpdir, "onnx")))
    result = reader.predict(query="Where does Paul live?", documents=[docs[0]])
    assert result["answers"][0].answer == "New York"
