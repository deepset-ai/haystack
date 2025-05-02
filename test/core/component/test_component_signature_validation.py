from typing import Dict, Any, List
import pytest
from haystack.core.component import component
from haystack.core.component.component import ComponentError


@component
class ValidComponent:
    def run(self, text: str) -> Dict[str, Any]:
        return {"result": text}

    async def run_async(self, text: str) -> Dict[str, Any]:
        return {"result": text}


@component
class DifferentParamNameComponent:
    def run(self, text: str) -> Dict[str, Any]:
        return {"result": text}

    async def run_async(self, input_text: str) -> Dict[str, Any]:
        return {"result": input_text}


@component
class DifferentParamTypeComponent:
    def run(self, text: str) -> Dict[str, Any]:
        return {"result": text}

    async def run_async(self, text: List[str]) -> Dict[str, Any]:
        return {"result": text[0]}


@component
class DifferentDefaultValueComponent:
    def run(self, text: str = "default") -> Dict[str, Any]:
        return {"result": text}

    async def run_async(self, text: str = "different") -> Dict[str, Any]:
        return {"result": text}


@component
class DifferentParamKindComponent:
    def run(self, text: str) -> Dict[str, Any]:
        return {"result": text}

    async def run_async(self, *, text: str) -> Dict[str, Any]:
        return {"result": text}


@component
class DifferentParamCountComponent:
    def run(self, text: str) -> Dict[str, Any]:
        return {"result": text}

    async def run_async(self, text: str, extra: str) -> Dict[str, Any]:
        return {"result": text + extra}


def test_valid_signatures():
    component = ValidComponent()
    assert component.run("test") == {"result": "test"}


def test_different_param_names():
    with pytest.raises(ComponentError, match="name mismatch"):
        DifferentParamNameComponent()


def test_different_param_types():
    with pytest.raises(ComponentError, match="type mismatch"):
        DifferentParamTypeComponent()


def test_different_default_values():
    with pytest.raises(ComponentError, match="default value mismatch"):
        DifferentDefaultValueComponent()


def test_different_param_kinds():
    with pytest.raises(ComponentError, match="kind \(POSITIONAL, KEYWORD, etc\.\) mismatch: "):
        DifferentParamKindComponent()


def test_different_param_count():
    with pytest.raises(ComponentError, match="Different number of parameters"):
        DifferentParamCountComponent()
