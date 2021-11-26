from haystack.nodes import FARMReader
import torch

def test_distillation():
    student = FARMReader(model_name_or_path="prajjwal1/bert-tiny")
    teacher = FARMReader(model_name_or_path="prajjwal1/bert-small")

    # create a checkpoint of weights before distillation
    student_weights = []
    for name, weight in student.inferencer.model.named_parameters():
        if "weight" in name and weight.requires_grad:
            student_weights.append(torch.clone(weight))

    assert len(student_weights) == 22

    student_weights.pop(-2) # pooler is not updated due to different attention head
    
    student.distil_from(teacher, data_dir="samples/squad", train_filename="tiny.json")

    # create new checkpoint
    new_student_weights = [torch.clone(param) for param in student.inferencer.model.parameters()]

    new_student_weights = []
    for name, weight in student.inferencer.model.named_parameters():
        if "weight" in name and weight.requires_grad:
            new_student_weights.append(weight)

    assert len(new_student_weights) == 22

    new_student_weights.pop(-2) # pooler is not updated due to different attention head

    # check if weights have changed
    assert not any(torch.equal(old_weight, new_weight) for old_weight, new_weight in zip(student_weights, new_student_weights))