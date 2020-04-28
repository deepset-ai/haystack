import pytest

from haystack.reader.farm import FARMReader


def test_farm_reader():
    reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2", use_gpu=False)
    assert reader is not None
    assert isinstance(reader, FARMReader)
