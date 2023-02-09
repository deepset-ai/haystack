"""
Script to perform data augmentation on a SQuAD like dataset to increase training data.
It follows the approach oultined in the TinyBERT paper: https://arxiv.org/pdf/1909.10351.pdf.
It takes a SQuAD like dataset as input and writes the augmented dataset to a new file in the same format.
There are no answer labels in the augmented dataset and every question is marked impossible.
This isn't a problem for distillation. However, this means that the augmented dataset is not suitable for training
a model on its own.
Usage:
    python augment_squad.py --squad_path <squad_path> --output_path <output_path> \
        --multiplication_factor <multiplication_factor> --word_possibilities <word_possibilities> \
        --replace_probability <replace_probability> --glove_path <glove_path> \
        --batch_size <batch_size> --device <device> --tokenizer <tokenizer> --model <model>
Arguments:
    squad_path: Path to the input dataset. Must have the same structure as the official squad json.
    output_path: Path to the output dataset.
    multiplication_factor: Number of times to augment the dataset.
    word_possibilities: Number of possible words to replace a word with.
    replace_probability: Probability of replacing a word with a different word.
    glove_path: Path to the GloVe vectors. If it does not exist, it will be downloaded.
    batch_size: Batch size for MLM model.
    device: Device to use for MLM model. Usually either "cpu:0" or "cuda:0".
    tokenizer: Huggingface tokenizer identifier.
    model: Huggingface MLM model identifier.
"""

from typing import Tuple, List, Union

import json
import random
import argparse
import logging
from pathlib import Path
from zipfile import ZipFile
from copy import copy, deepcopy

import torch
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import requests
import numpy as np
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def load_glove(
    glove_path: Path = Path("glove.txt"),
    vocab_size: int = 100_000,
    device: Union[str, torch.device] = torch.device("cpu:0"),
    timeout: Union[float, Tuple[float, float]] = 10.0,
) -> Tuple[dict, dict, torch.Tensor]:
    """
    Loads the GloVe vectors and returns a mapping from words to their GloVe vector indices and the other way around.
    :param timeout: How many seconds to wait for the server to send data before giving up,
        as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
        Defaults to 10 seconds.
    """

    if not glove_path.exists():  # download and extract glove if necessary
        logger.info("Provided glove file not found. Downloading it instead.")
        glove_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path = glove_path.parent / (glove_path.name + ".zip")
        request = requests.get(
            "https://nlp.stanford.edu/data/glove.42B.300d.zip", allow_redirects=True, timeout=timeout
        )
        with zip_path.open("wb") as downloaded_file:
            downloaded_file.write(request.content)
        with ZipFile(zip_path, "r") as zip_file:
            glove_file = zip_file.namelist()[0]
            with glove_path.open("wb") as g:
                g.write(zip_file.read(glove_file))

    word_id_mapping = {}
    id_word_mapping = {}
    vector_list = []
    with open(glove_path, "r") as f:
        for i, line in enumerate(f):
            if i == vocab_size:  # limit vocab size
                break
            split = line.split()
            word_id_mapping[split[0]] = i
            id_word_mapping[i] = split[0]
            vector_list.append(torch.tensor([float(x) for x in split[1:]]))
    vectors = torch.stack(vector_list)
    with torch.inference_mode():
        vectors = vectors.to(device)
        vectors = F.normalize(vectors, dim=1)
    return word_id_mapping, id_word_mapping, vectors


def tokenize_and_extract_words(text: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[torch.Tensor, List[str], dict]:
    # tokenizes text and returns, in addition to the tokens, indices and mapping of the words that were not split into subwords
    # this is important as MLM is not used for subwords
    words = tokenizer.basic_tokenizer.tokenize(text)

    subwords = [tokenizer.wordpiece_tokenizer.tokenize(word) for word in words]

    word_subword_mapping = {}

    j = 0
    for i, subwords_ in enumerate(subwords):
        j += len(subwords_)
        if j >= 510:  # sequence length may not be longer than 512 (1 cls + 510 tokens + 1 sep)
            break
        if len(subwords_) == 1:
            word_subword_mapping[i] = j

    subwords = [subword for subwords_ in subwords for subword in subwords_]  # flatten list of lists

    input_ids = tokenizer.convert_tokens_to_ids(subwords[:510])
    input_ids.insert(0, tokenizer.cls_token_id)
    input_ids.append(tokenizer.sep_token_id)

    return input_ids, words, word_subword_mapping


def get_replacements(
    glove_word_id_mapping: dict,
    glove_id_word_mapping: dict,
    glove_vectors: np.ndarray,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    word_possibilities: int = 20,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu:0"),
) -> List[List[str]]:
    """Returns a list of possible replacements for each word in the text."""
    input_ids, words, word_subword_mapping = tokenize_and_extract_words(text, tokenizer)

    # masks words which were not split into subwords by the tokenizer
    inputs = []
    for subword_index in word_subword_mapping.values():
        input_ids_ = copy(input_ids)
        input_ids_[subword_index] = tokenizer.mask_token_id
        inputs.append((input_ids_, subword_index))

    # doing batched forward pass
    with torch.inference_mode():
        prediction_list = []
        while len(inputs) != 0:
            batch_list, token_indices = tuple(zip(*inputs[:batch_size]))
            batch = torch.tensor(batch_list)
            batch = batch.to(device)
            logits = model(input_ids=batch)["logits"]
            relevant_logits = logits[torch.arange(batch.shape[0]), token_indices]
            ranking = torch.topk(relevant_logits, word_possibilities, dim=1)
            prediction_list.append(ranking.indices.cpu())
            inputs = inputs[batch_size:]
        predictions = torch.cat(prediction_list, dim=0)

    # creating list of possible replacements for each word
    possible_words = []

    batch_index = 0
    for i, word in enumerate(words):
        if i in word_subword_mapping:  # word was not split into subwords so we can use MLM output
            subword_index = word_subword_mapping[i]
            ranking = predictions[batch_index]
            possible_words_ = [word]
            for token in ranking:
                word = tokenizer.convert_ids_to_tokens([token])[0]
                if not word.startswith("##"):
                    possible_words_.append(word)

            possible_words.append(possible_words_)

            batch_index += 1
        elif word in glove_word_id_mapping:  # word was split into subwords so we use glove instead
            word_id = glove_word_id_mapping[word]
            glove_vector = glove_vectors[word_id]
            with torch.inference_mode():
                word_similarities = torch.mm(glove_vectors, glove_vector.unsqueeze(1)).squeeze(1)  # type: ignore [arg-type]
                ranking = torch.argsort(word_similarities, descending=True)[: word_possibilities + 1]
                possible_words.append([glove_id_word_mapping[int(id_)] for id_ in ranking.cpu()])
        else:  # word was not in glove either so we can't find any replacements
            possible_words.append([word])

    return possible_words


def augment(
    word_id_mapping: dict,
    id_word_mapping: dict,
    vectors: np.ndarray,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    multiplication_factor: int = 20,
    word_possibilities: int = 20,
    replace_probability: float = 0.4,
    batch_size: int = 16,
    device: Union[str, torch.device] = torch.device("cpu:0"),
) -> List[str]:
    device = torch.device(device)
    # returns a list of different augmented versions of the text
    replacements = get_replacements(
        glove_word_id_mapping=word_id_mapping,
        glove_id_word_mapping=id_word_mapping,
        glove_vectors=vectors,
        model=model,
        tokenizer=tokenizer,
        text=text,
        word_possibilities=word_possibilities,
        batch_size=batch_size,
        device=device,
    )
    new_texts = []
    for _ in range(multiplication_factor):
        new_text = []
        for possible_words in replacements:
            if len(possible_words) == 1:
                new_text.append(possible_words[0])
                continue
            if random.random() < replace_probability:
                new_text.append(random.choice(possible_words[1:]))
            else:
                new_text.append(possible_words[0])
        new_texts.append(" ".join(new_text))
    return new_texts


def augment_squad(
    squad_path: Path,
    output_path: Path,
    glove_path: Path = Path("glove.txt"),
    model: str = "bert-base-uncased",
    tokenizer: str = "bert-base-uncased",
    multiplication_factor: int = 20,
    word_possibilities: int = 20,
    replace_probability: float = 0.4,
    device: Union[str, torch.device] = "cpu:0",
    batch_size: int = 16,
):
    """Loads a squad dataset, augments the contexts, and saves the result in SQuAD format."""
    device = torch.device(device)
    # loading model and tokenizer
    transformers_model = AutoModelForMaskedLM.from_pretrained(model)
    transformers_model.to(device)
    transformers_tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    # load glove for words that do not have one distinct token, but are split into subwords
    word_id_mapping, id_word_mapping, vectors = load_glove(glove_path=glove_path, device=device)

    # load squad dataset
    with open(squad_path, "r") as f:
        squad = json.load(f)

    topics = []

    for topic in tqdm(squad["data"]):
        paragraphs = []
        for paragraph in topic["paragraphs"]:
            # make every question unanswerable as answer strings will probably not match and aren't relevant for distillation
            for question in paragraph["qas"]:
                question["answers"] = []
            context = paragraph["context"]
            contexts = augment(
                word_id_mapping=word_id_mapping,
                id_word_mapping=id_word_mapping,
                vectors=vectors,  # type: ignore [arg-type]
                model=transformers_model,
                tokenizer=transformers_tokenizer,
                text=context,
                multiplication_factor=multiplication_factor,
                word_possibilities=word_possibilities,
                replace_probability=replace_probability,
                device=device,
                batch_size=batch_size,
            )
            paragraphs_ = []
            for context in contexts:
                new_paragraph = deepcopy(paragraph)
                new_paragraph["context"] = context
                paragraphs_.append(new_paragraph)
            paragraphs += paragraphs_
        topic["paragraphs"] = paragraphs
        topics.append(topic)
    squad["topics"] = topics

    # save new dataset
    with open(output_path, "w") as f:
        json.dump(squad, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_path", type=Path, required=True, help="Path to the squad json file")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save augmented dataaset")
    parser.add_argument(
        "--multiplication_factor", type=int, default=20, help="Factor by which dataset size is multiplied"
    )
    parser.add_argument(
        "--word_possibilities",
        type=int,
        default=5,
        help="Number of possible words to choose from when replacing a word",
    )
    parser.add_argument("--replace_probability", type=float, default=0.4, help="Probability of replacing a word")
    parser.add_argument("--glove_path", type=Path, default="glove.txt", help="Path to the glove file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for forward pass")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="Huggingface model identifier for MLM model"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="bert-base-uncased", help="Huggingface tokenizer identifier for MLM model"
    )

    args = parser.parse_args()

    augment_squad(**vars(args))
