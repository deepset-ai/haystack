import pytest

def validate_pipeline_output(output: dict, expected_keys: set) -> bool:
    if not isinstance(output, dict):
        raise TypeError("Output must be a dictionary")
    missing = expected_keys - set(output.keys())
    if missing:
        raise KeyError(f"Missing required output keys: {missing}")
    return True

def test_valid_output():
    out = {"documents": [], "meta": {}}
    assert validate_pipeline_output(out, {"documents", "meta"}) is True

def test_missing_output_keys():
    out = {"documents": []}
    with pytest.raises(KeyError, match="Missing"):
        validate_pipeline_output(out, {"documents", "meta"})
