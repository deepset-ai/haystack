import logging
from pathlib import Path

from transformers import AutoTokenizer

from haystack.modeling.data_handler.processor import SquadProcessor
from haystack.modeling.utils import set_all_seeds
import torch


def test_processor_saving_loading(tmp_path, caplog, samples_path):
    if caplog is not None:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    lang_model = "roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=lang_model, do_lower_case=False)

    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=256,
        label_list=["start_token", "end_token"],
        train_filename="train-sample.json",
        dev_filename="dev-sample.json",
        test_filename=None,
        data_dir=samples_path / "qa",
    )

    dicts = processor.file_to_dicts(file=samples_path / "qa" / "dev-sample.json")
    data, tensor_names, _ = processor.dataset_from_dicts(dicts=dicts, indices=[1])

    save_dir = tmp_path / Path("testsave/processor")
    processor.save(save_dir)

    processor = processor.load_from_dir(save_dir)
    dicts = processor.file_to_dicts(file=samples_path / "qa" / "dev-sample.json")
    data_loaded, tensor_names_loaded, _ = processor.dataset_from_dicts(dicts, indices=[1])

    assert tensor_names == tensor_names_loaded
    for i in range(len(data.tensors)):
        assert torch.all(torch.eq(data.tensors[i], data_loaded.tensors[i]))


if __name__ == "__main__":
    test_processor_saving_loading(None)
