import numpy as np
import pytest
import pandas as pd
from pathlib import Path

from haystack.utils.preprocessing import convert_files_to_docs, tika_convert_files_to_docs
from haystack.utils.cleaning import clean_wiki_text
from haystack.utils.augment_squad import augment_squad
from haystack.utils.squad_data import SquadData
from haystack.utils.context_matching import calculate_context_similarity, match_context, match_contexts

from ..conftest import SAMPLES_PATH

TEST_CONTEXT = context = """Der Merkantilismus förderte Handel und Verkehr mit teils marktkonformen, teils dirigistischen Maßnahmen. 
An der Schwelle zum 19. Jahrhundert entstand ein neuer Typus des Nationalstaats, der die Säkularisation durchsetzte, 
moderne Bildungssysteme etablierte und die Industrialisierung vorantrieb.\n
Beim Begriff der Aufklärung geht es auch um die Prozesse zwischen diesen frühneuzeitlichen Eckpunkten. 
Man versucht die fortschrittlichen Faktoren zu definieren, die in das 19. Jahrhundert führten. 
Widerstände gegen diesen Fortschritt werden anti-aufklärerischen Kräften oder unreflektierten Traditionen zugeordnet. 
Die Epochendefinition rückt vor allem publizistisch tätige Gruppen in den gesellschaftlichen Fokus, 
die zunächst selten einen bürgerlichen Hintergrund aufwiesen, sondern weitaus häufiger der Geistlichkeit oder Aristokratie angehörten: 
Wissenschaftler, Journalisten, Autoren, sogar Regenten, die Traditionen der Kritik unterzogen, indem sie sich auf die Vernunftperspektive beriefen."""


TEST_CONTEXT_2 = """Beer is one of the oldest[1][2][3] and most widely consumed[4] alcoholic drinks in the world, and the third most popular drink overall after water and tea.[5] It is produced by the brewing and fermentation of starches, mainly derived from cereal grains—most commonly from malted barley, though wheat, maize (corn), rice, and oats are also used. During the brewing process, fermentation of the starch sugars in the wort produces ethanol and carbonation in the resulting beer.[6] Most modern beer is brewed with hops, which add bitterness and other flavours and act as a natural preservative and stabilizing agent. Other flavouring agents such as gruit, herbs, or fruits may be included or used instead of hops. In commercial brewing, the natural carbonation effect is often removed during processing and replaced with forced carbonation.[7]
Some of humanity's earliest known writings refer to the production and distribution of beer: the Code of Hammurabi included laws regulating beer and beer parlours,[8] and "The Hymn to Ninkasi", a prayer to the Mesopotamian goddess of beer, served as both a prayer and as a method of remembering the recipe for beer in a culture with few literate people.[9][10]
Beer is distributed in bottles and cans and is also commonly available on draught, particularly in pubs and bars. The brewing industry is a global business, consisting of several dominant multinational companies and many thousands of smaller producers ranging from brewpubs to regional breweries. The strength of modern beer is usually around 4% to 6% alcohol by volume (ABV), although it may vary between 0.5% and 20%, with some breweries creating examples of 40% ABV and above.[11]
Beer forms part of the culture of many nations and is associated with social traditions such as beer festivals, as well as a rich pub culture involving activities like pub crawling, pub quizzes and pub games.
When beer is distilled, the resulting liquor is a form of whisky.[12]
"""


def test_convert_files_to_docs():
    documents = convert_files_to_docs(
        dir_path=(SAMPLES_PATH).absolute(), clean_func=clean_wiki_text, split_paragraphs=True
    )
    assert documents and len(documents) > 0


@pytest.mark.tika
def test_tika_convert_files_to_docs():
    documents = tika_convert_files_to_docs(dir_path=SAMPLES_PATH, clean_func=clean_wiki_text, split_paragraphs=True)
    assert documents and len(documents) > 0


def test_squad_augmentation():
    input_ = SAMPLES_PATH / "squad" / "tiny.json"
    output = SAMPLES_PATH / "squad" / "tiny_augmented.json"
    glove_path = SAMPLES_PATH / "glove" / "tiny.txt"  # dummy glove file, will not even be use when augmenting tiny.json
    multiplication_factor = 5
    augment_squad(
        model="distilbert-base-uncased",
        tokenizer="distilbert-base-uncased",
        squad_path=input_,
        output_path=output,
        glove_path=glove_path,
        multiplication_factor=multiplication_factor,
    )
    original_squad = SquadData.from_file(input_)
    augmented_squad = SquadData.from_file(output)
    assert original_squad.count(unit="paragraph") == augmented_squad.count(unit="paragraph") * multiplication_factor


def test_squad_to_df():
    df = pd.DataFrame(
        [["title", "context", "question", "id", "answer", 1, False]],
        columns=["title", "context", "question", "id", "answer_text", "answer_start", "is_impossible"],
    )

    expected_result = [
        {
            "title": "title",
            "paragraphs": [
                {
                    "context": "context",
                    "qas": [
                        {
                            "question": "question",
                            "id": "id",
                            "answers": [{"text": "answer", "answer_start": 1}],
                            "is_impossible": False,
                        }
                    ],
                }
            ],
        }
    ]

    result = SquadData.df_to_data(df)

    assert result == expected_result


def test_calculate_context_similarity_on_parts_of_whole_document():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size]
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 100.0


def test_calculate_context_similarity_on_parts_of_whole_document_different_case():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size].lower()
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 100.0


def test_calculate_context_similarity_on_parts_of_whole_document_different_whitesapce():
    whole_document = TEST_CONTEXT
    words = whole_document.split()
    min_length = 100
    context_word_size = 20
    for i in range(len(words) - context_word_size):
        partial_context = "\n\t\t\t".join(words[i : i + context_word_size])
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 100.0


def test_calculate_context_similarity_min_length():
    whole_document = TEST_CONTEXT
    min_length = 100
    context_size = min_length - 1
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size]
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 0.0


def test_calculate_context_similarity_on_partially_overlapping_contexts():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    stride = context_size // 2
    for i in range(len(whole_document) - context_size - stride):
        partial_context_1 = whole_document[i : i + context_size]
        partial_context_2 = whole_document[i + stride : i + stride + context_size]
        score = calculate_context_similarity(partial_context_1, partial_context_2, min_length=min_length)
        assert score >= 65.0


def test_calculate_context_similarity_on_non_matching_contexts():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    scores = []
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size // 2] + _get_random_chars(context_size // 2)
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        scores.append(score)
    for i in range(len(whole_document) - context_size):
        partial_context = (
            _get_random_chars(context_size // 2) + whole_document[i + context_size // 2 : i + context_size]
        )
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        scores.append(score)
    accuracy = np.where(np.array(scores) < 65, 1, 0).mean()
    assert accuracy > 0.99


def test_calculate_context_similarity_on_parts_of_whole_document_with_noise():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = _insert_noise(whole_document[i : i + context_size], 0.1)
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score >= 85.0


def test_calculate_context_similarity_on_partially_overlapping_contexts_with_noise():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    stride = context_size // 2
    scores = []
    for i in range(len(whole_document) - context_size - stride):
        partial_context_1 = whole_document[i : i + context_size]
        partial_context_2 = _insert_noise(whole_document[i + stride : i + stride + context_size], 0.1)
        score = calculate_context_similarity(partial_context_1, partial_context_2, min_length=min_length)
        scores.append(score)
    accuracy = np.where(np.array(scores) >= 65, 1, 0).mean()
    assert accuracy > 0.99


def test_match_context():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size]
        candidates = ((str(i), TEST_CONTEXT if i == 0 else TEST_CONTEXT_2) for i in range(10))
        results = match_context(partial_context, candidates, min_length=min_length, num_processes=2)
        assert len(results) == 1
        id, score = results[0]
        assert id == "0"
        assert score == 100.0


def test_match_context_single_process():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size]
        candidates = ((str(i), TEST_CONTEXT if i == 0 else TEST_CONTEXT_2) for i in range(10))
        results = match_context(partial_context, candidates, min_length=min_length, num_processes=1)
        assert len(results) == 1
        id, score = results[0]
        assert id == "0"
        assert score == 100.0


def test_match_contexts():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    candidates = ((str(i), TEST_CONTEXT if i == 0 else TEST_CONTEXT_2) for i in range(10))
    partial_contexts = [whole_document[i : i + context_size] for i in range(len(whole_document) - context_size)]
    result_list = match_contexts(partial_contexts, candidates, min_length=min_length, num_processes=2)
    assert len(result_list) == len(partial_contexts)
    for results in result_list:
        assert len(results) == 1
        id, score = results[0]
        assert id == "0"
        assert score == 100.0


def _get_random_chars(size: int):
    chars = np.random.choice(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZß?/.,;:-#äöüÄÖÜ+*~1234567890$€%&!§ "), size=size
    )
    return "".join(list(chars))


def _insert_noise(input: str, ratio):
    size = int(ratio * len(input))
    insert_idxs = sorted(np.random.choice(range(len(input)), size=size, replace=False), reverse=True)
    insert_chars = np.random.choice(
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZß?/.,;:-#äöüÄÖÜ+*~1234567890$€%&!§"), size=size
    )
    for idx, char in zip(insert_idxs, insert_chars):
        input = input[:idx] + char + input[idx:]
    return input
