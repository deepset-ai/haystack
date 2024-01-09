import pytest

from haystack import ComponentError, DeserializationError
from haystack.components.extractors import NamedEntityExtractor, NamedEntityExtractorBackend


@pytest.mark.unit
def test_named_entity_extractor_backend():
    _ = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER")

    _ = NamedEntityExtractor(backend="hugging_face", model_name_or_path="dslim/bert-base-NER")

    _ = NamedEntityExtractor(backend=NamedEntityExtractorBackend.SPACY, model_name_or_path="en_core_web_sm")

    _ = NamedEntityExtractor(backend="spacy", model_name_or_path="en_core_web_sm")

    with pytest.raises(ComponentError, match=r"Invalid backend"):
        NamedEntityExtractor(backend="random_backend", model_name_or_path="dslim/bert-base-NER")


@pytest.mark.unit
def test_named_entity_extractor_serde():
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER", device_id=-1
    )

    serde_data = extractor.to_dict()
    new_extractor = NamedEntityExtractor.from_dict(serde_data)

    assert type(new_extractor._backend) == type(extractor._backend)
    assert new_extractor._backend.model_name == extractor._backend.model_name
    assert new_extractor._backend.device_id == extractor._backend.device_id

    with pytest.raises(DeserializationError, match=r"Couldn't deserialize"):
        serde_data["init_parameters"].pop("backend")
        _ = NamedEntityExtractor.from_dict(serde_data)
