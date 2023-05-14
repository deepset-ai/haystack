import torch

from haystack.nodes import FARMReader
from haystack.modeling.data_handler.processor import UnlabeledTextProcessor


def create_checkpoint(model):
    weights = []
    for name, weight in model.inferencer.model.named_parameters():
        if "weight" in name and weight.requires_grad:
            weights.append(torch.clone(weight))
    return weights


def assert_weight_change(weights, new_weights):
    print([torch.equal(old_weight, new_weight) for old_weight, new_weight in zip(weights, new_weights)])
    assert not any(torch.equal(old_weight, new_weight) for old_weight, new_weight in zip(weights, new_weights))


def test_prediction_layer_distillation(samples_path):
    student = FARMReader(model_name_or_path="prajjwal1/bert-mini", num_processes=0)
    teacher = FARMReader(model_name_or_path="prajjwal1/bert-small", num_processes=0)

    # create a checkpoint of weights before distillation
    student_weights = create_checkpoint(student)

    assert len(student_weights) == 38

    student_weights.pop(-2)  # pooler is not updated due to different attention head

    student.distil_prediction_layer_from(teacher, data_dir=samples_path / "squad", train_filename="tiny.json")

    # create new checkpoint
    new_student_weights = create_checkpoint(student)

    assert len(new_student_weights) == 38

    new_student_weights.pop(-2)  # pooler is not updated due to different attention head

    # check if weights have changed
    assert_weight_change(student_weights, new_student_weights)


def test_intermediate_layer_distillation(samples_path):
    student = FARMReader(model_name_or_path="huawei-noah/TinyBERT_General_4L_312D")
    teacher = FARMReader(model_name_or_path="bert-base-uncased")

    # create a checkpoint of weights before distillation
    student_weights = create_checkpoint(student)

    assert len(student_weights) == 38

    student_weights.pop(-1)  # last layer is not affected by tinybert loss
    student_weights.pop(-1)  # pooler is not updated due to different attention head

    student.distil_intermediate_layers_from(
        teacher_model=teacher, data_dir=samples_path / "squad", train_filename="tiny.json"
    )

    # create new checkpoint
    new_student_weights = create_checkpoint(student)

    assert len(new_student_weights) == 38

    new_student_weights.pop(-1)  # last layer is not affected by tinybert loss
    new_student_weights.pop(-1)  # pooler is not updated due to different attention head

    # check if weights have changed
    assert_weight_change(student_weights, new_student_weights)


def test_intermediate_layer_distillation_from_scratch(samples_path):
    student = FARMReader(model_name_or_path="huawei-noah/TinyBERT_General_4L_312D")
    teacher = FARMReader(model_name_or_path="bert-base-uncased")

    # create a checkpoint of weights before distillation
    student_weights = create_checkpoint(student)

    assert len(student_weights) == 38

    student_weights.pop(-1)  # last layer is not affected by tinybert loss
    student_weights.pop(-1)  # pooler is not updated due to different attention head

    processor = UnlabeledTextProcessor(
        tokenizer=teacher.inferencer.processor.tokenizer,
        max_seq_len=128,
        train_filename="doc_2.txt",
        data_dir=samples_path / "docs",
    )
    student.distil_intermediate_layers_from(
        teacher_model=teacher, data_dir=samples_path / "squad", train_filename="tiny.json", processor=processor
    )

    # create new checkpoint
    new_student_weights = create_checkpoint(student)

    assert len(new_student_weights) == 38

    new_student_weights.pop(-1)  # last layer is not affected by tinybert loss
    new_student_weights.pop(-1)  # pooler is not updated due to different attention head

    # check if weights have changed
    assert_weight_change(student_weights, new_student_weights)
