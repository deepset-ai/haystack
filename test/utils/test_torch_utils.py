import pytest
import torch

from haystack.utils.torch_utils import extract_torch_dtype


def test_extract_torch_dtype() -> None:
    torch_dtype = extract_torch_dtype(**{"torch_dtype": torch.float16})
    assert torch_dtype == torch.float16


def test_extract_torch_dtype_none() -> None:
    torch_dtype = extract_torch_dtype(**{})
    assert torch_dtype is None


def test_extract_torch_dtype_str() -> None:
    torch_dtype = extract_torch_dtype(**{"torch_dtype": "torch.float16"})
    assert torch_dtype == torch.float16


def test_extract_torch_dtype_auto() -> None:
    torch_dtype = extract_torch_dtype(**{"torch_dtype": "auto"})
    assert torch_dtype == "auto"


def test_extract_torch_dtype_invalid() -> None:
    with pytest.raises(ValueError):
        _ = extract_torch_dtype(**{"torch_dtype": "random string"})
