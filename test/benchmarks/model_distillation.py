from haystack.nodes import FARMReader
import json
import os
import requests
from pathlib import Path

from haystack.nodes import FARMReader

from typing import Union
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

download_links = {
    "squad2": {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "test": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    },
    "squad": {
        "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "test": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    }
}

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def download_file(url: str, path: Path):
    request = requests.get(url, allow_redirects=True)
    with path.open("wb") as f:
        f.write(request.content)

def download_dataset(dataset: Union[dict, str], download_folder: Path):
    train_file = "train.json"
    test_file = "test.json"
    if download_folder.exists():
        assert download_folder.is_dir()
        if (download_folder/train_file).is_file() and (download_folder/test_file).is_file():
            return train_file, test_file
    if type(dataset) is not dict:
        dataset = download_links[dataset]
    train = dataset["train"]
    test = dataset["test"]
    download_folder.mkdir(parents=True, exist_ok=True)
    download_file(train, download_folder/train_file)
    download_file(test, download_folder/test_file)
    return train_file, test_file

def eval(model: FARMReader, download_folder: Path, test_file: str):
    return model.eval_on_file(data_dir=download_folder, test_filename=test_file)

def train_student(model_name: str, download_folder: Path, train_file: str, test_file: str, epochs: int, batch_size: int) -> dict:
    model = FARMReader(model_name_or_path=model_name)
    model.train(data_dir=download_folder, train_filename=train_file, n_epochs=epochs, batch_size=batch_size, caching=True)
    return eval(model, download_folder, test_file)

def train_student_with_distillation(student_name: str, teacher_name: str, download_folder: Path, train_file: str, test_file: str,
epochs: int, student_batch_size: int, teacher_batch_size: int) -> dict:
    student = FARMReader(model_name_or_path=student_name)
    teacher = FARMReader(model_name_or_path=teacher_name)
    student.distil_from(teacher, data_dir=download_folder, train_filename=train_file, n_epochs=epochs, caching=True,
    student_batch_size=student_batch_size, teacher_batch_size=teacher_batch_size)
    return eval(student, download_folder, test_file)

def main():
    config = load_config("distillation_config.json")
    download_folder = Path(__file__).parent.resolve()/Path(config["download_folder"])
    student = config["student_model"]
    teacher = config["teacher_model"]

    logger.info("Downloading dataset")
    train_file, test_file = download_dataset(config["dataset"], download_folder)

    logger.info("Training student without distillation as a baseline")
    results_student = train_student(student["model_name_or_path"], download_folder, train_file, test_file, config["epochs"], student["batch_size"])

    logger.info("Training student with distillation")
    results_student_with_distillation = train_student_with_distillation(student["model_name_or_path"], teacher["model_name_or_path"], download_folder,
    train_file, test_file, config["epochs"], student["batch_size"], teacher["batch_size"])

    logger.info("Evaluating teacher")
    results_teacher = eval(FARMReader(model_name_or_path=teacher["model_name_or_path"]), download_folder, test_file)

    logger.info("Evaluation results:")
    descriptions = ["Results of teacher", "Results of student without distillation (baseline)", "Results of student with distillation"]
    for evaluation, description in zip([results_teacher, results_student, results_student_with_distillation], descriptions):
        logger.info(description)
        logger.info(f"EM: {evaluation['EM']}")
        logger.info(f"F1: {evaluation['f1']}")
        logger.info(f"Top n accuracy: {evaluation['top_n_accuracy']}")



if __name__ == "__main__":
    main()