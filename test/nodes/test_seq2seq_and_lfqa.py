import pytest

from haystack.nodes.answer_generator import Seq2SeqGenerator


@pytest.mark.unit
def test_seq2seq_unknown_converter():
    seq2seq = Seq2SeqGenerator(model_name_or_path="patrickvonplaten/t5-tiny-random")
    with pytest.raises(Exception, match="doesn't have input converter registered for patrickvonplaten/t5-tiny-random"):
        seq2seq.predict(query="irrelevant")


@pytest.mark.unit
def test_seq2seq_invalid_converter():
    class _InvalidConverter:
        def __call__(self, some_invalid_para: str, another_invalid_param: str) -> None:
            pass

    seq2seq = Seq2SeqGenerator(
        model_name_or_path="patrickvonplaten/t5-tiny-random", input_converter=_InvalidConverter()
    )
    with pytest.raises(Exception, match="does not have a valid __call__ method signature"):
        seq2seq.predict(query="irrelevant")
