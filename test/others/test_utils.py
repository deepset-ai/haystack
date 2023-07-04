import importlib
import logging
from typing import List
from unittest import mock

import numpy as np
import pytest
import pandas as pd
import responses
from responses import matchers

import _pytest

from haystack.schema import Answer, Document, Span, Label
from haystack.utils import print_answers
from haystack.utils.deepsetcloud import DeepsetCloud, DeepsetCloudExperiments
from haystack.utils.import_utils import get_filename_extension_from_url
from haystack.utils.labels import aggregate_labels
from haystack.utils.preprocessing import convert_files_to_docs, tika_convert_files_to_docs
from haystack.utils.cleaning import clean_wiki_text
from haystack.utils.context_matching import calculate_context_similarity, match_context, match_contexts

from .. import conftest
from ..conftest import DC_API_ENDPOINT, DC_API_KEY, MOCK_DC, deepset_cloud_fixture, fail_at_version

TEST_CONTEXT = """Der Merkantilismus förderte Handel und Verkehr mit teils marktkonformen, teils dirigistischen Maßnahmen.
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


# Util function for testing
def noop():
    return True


@pytest.mark.unit
def test_deprecation_previous_major_and_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        with pytest.warns(match="This feature is marked for removal in v1.1"):
            fail_at_version(1, 1)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(1, 1)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(1, 1)(noop)()


@pytest.mark.unit
def test_deprecation_previous_major_same_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        with pytest.warns(match="This feature is marked for removal in v1.2"):
            fail_at_version(1, 2)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(1, 2)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(1, 2)(noop)()


@pytest.mark.unit
def test_deprecation_previous_major_later_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        with pytest.warns(match="This feature is marked for removal in v1.3"):
            fail_at_version(1, 3)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(1, 3)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(1, 3)(noop)()


@pytest.mark.unit
def test_deprecation_same_major_previous_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        with pytest.warns(match="This feature is marked for removal in v2.1"):
            fail_at_version(2, 1)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(2, 1)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(2, 1)(noop)()


@pytest.mark.unit
def test_deprecation_same_major_same_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        with pytest.warns(match="This feature is marked for removal in v2.2"):
            fail_at_version(2, 2)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(2, 2)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        with pytest.raises(_pytest.outcomes.Failed):
            fail_at_version(2, 2)(noop)()


@pytest.mark.unit
def test_deprecation_same_major_later_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        assert fail_at_version(2, 3)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        assert fail_at_version(2, 3)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        assert fail_at_version(2, 3)(noop)()


@pytest.mark.unit
def test_deprecation_later_major_previous_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        assert fail_at_version(3, 1)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        assert fail_at_version(3, 1)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        assert fail_at_version(3, 1)(noop)()


@pytest.mark.unit
def test_deprecation_later_major_same_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        assert fail_at_version(3, 2)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        assert fail_at_version(3, 2)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        assert fail_at_version(3, 2)(noop)()


@pytest.mark.unit
def test_deprecation_later_major_later_minor():
    with mock.patch.object(conftest, "haystack_version", "2.2.2-rc0"):
        assert fail_at_version(3, 3)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2rc1"):
        assert fail_at_version(3, 3)(noop)()

    with mock.patch.object(conftest, "haystack_version", "2.2.2"):
        assert fail_at_version(3, 3)(noop)()


# Takes awhile to run
def test_convert_files_to_docs(samples_path):
    documents = convert_files_to_docs(
        dir_path=(samples_path).absolute(), clean_func=clean_wiki_text, split_paragraphs=True
    )
    assert documents and len(documents) > 0


@pytest.mark.unit
def test_get_filename_extension_from_url_without_params_zip():
    url = "http://www.mysite.com/resources/myfile.zip"
    file_name, extension = get_filename_extension_from_url(url)
    assert extension == "zip"
    assert file_name == "myfile"


@pytest.mark.unit
def test_get_filename_extension_from_url_with_params_zip():
    url = "https:/<S3_BUCKET_NAME>.s3.<REGION>.amazonaws.com/filename.zip?response-content-disposition=inline&X-Amz-Security-Token=<TOKEN>&X-Amz-Algorithm=<X-AMZ-ALGORITHM>&X-Amz-Date=<X-AMZ-DATE>&X-Amz-SignedHeaders=host&X-Amz-Expires=<X-AMZ-EXPIRES>&X-Amz-Credential=<CREDENTIAL>&X-Amz-Signature=<SIGNATURE>"
    file_name, extension = get_filename_extension_from_url(url)
    assert extension == "zip"
    assert file_name == "filename"


@pytest.mark.unit
def test_get_filename_extension_from_url_with_params_xz():
    url = "https:/<S3_BUCKET_NAME>.s3.<REGION>.amazonaws.com/filename.tar.xz?response-content-disposition=inline&X-Amz-Security-Token=<TOKEN>&X-Amz-Algorithm=<X-AMZ-ALGORITHM>&X-Amz-Date=<X-AMZ-DATE>&X-Amz-SignedHeaders=host&X-Amz-Expires=<X-AMZ-EXPIRES>&X-Amz-Credential=<CREDENTIAL>&X-Amz-Signature=<SIGNATURE>"
    file_name, extension = get_filename_extension_from_url(url)
    assert extension == "xz"
    assert file_name == "filename.tar"


@pytest.mark.tika
def test_tika_convert_files_to_docs(samples_path):
    documents = tika_convert_files_to_docs(dir_path=samples_path, clean_func=clean_wiki_text, split_paragraphs=True)
    assert documents and len(documents) > 0


@pytest.mark.unit
def test_calculate_context_similarity_on_parts_of_whole_document():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size]
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 100.0


@pytest.mark.unit
def test_calculate_context_similarity_on_parts_of_whole_document_different_case():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size].lower()
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 100.0


@pytest.mark.unit
def test_calculate_context_similarity_on_parts_of_whole_document_different_whitesapce():
    whole_document = TEST_CONTEXT
    words = whole_document.split()
    min_length = 100
    context_word_size = 20
    for i in range(len(words) - context_word_size):
        partial_context = "\n\t\t\t".join(words[i : i + context_word_size])
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 100.0


@pytest.mark.unit
def test_calculate_context_similarity_min_length():
    whole_document = TEST_CONTEXT
    min_length = 100
    context_size = min_length - 1
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size]
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score == 0.0


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_calculate_context_similarity_on_parts_of_whole_document_with_noise():
    whole_document = TEST_CONTEXT
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = _insert_noise(whole_document[i : i + context_size], 0.1)
        score = calculate_context_similarity(partial_context, whole_document, min_length=min_length)
        assert score >= 85.0


@pytest.mark.unit
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


def test_match_context_multi_process():
    whole_document = TEST_CONTEXT[:300]
    min_length = 100
    margin = 5
    context_size = min_length + margin
    for i in range(len(whole_document) - context_size):
        partial_context = whole_document[i : i + context_size]
        candidates = ((str(i), TEST_CONTEXT if i == 0 else TEST_CONTEXT_2) for i in range(1000))
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


def test_match_contexts_multi_process():
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


def test_secure_model_loading(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

    # now testing if just importing haystack is enough to enable secure loading of pytorch models
    import haystack

    importlib.reload(haystack)
    assert "already set to" in caplog.text


class TestAggregateLabels:
    @pytest.fixture
    def standard_labels(self) -> List[Label]:
        return [
            Label(
                id="standard",
                query="question",
                answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some", id="123"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
            # same label should be ignored
            Label(
                id="standard",
                query="question",
                answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some", id="123"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
            # different answer in same doc
            Label(
                id="diff-answer-same-doc",
                query="question",
                answer=Answer(answer="answer2", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some", id="123"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
            # answer in different doc
            Label(
                id="diff-answer-diff-doc",
                query="question",
                answer=Answer(answer="answer3", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some other", id="333"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
            # no_answer
            Label(
                id="no-answer",
                query="question",
                answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
                document=Document(content="some", id="777"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
            # no_answer in doc with other answer
            Label(
                id="no-answer-of-doc-with-other-answer",
                query="question",
                answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
                document=Document(content="some", id="123"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
            ),
            # negative label
            Label(
                id="negative",
                query="question",
                answer=Answer(answer="answer5", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some", id="123"),
                is_correct_answer=False,
                is_correct_document=True,
                origin="gold-label",
            ),
        ]

    @pytest.fixture
    def filter_meta_labels(self) -> List[Label]:
        return [
            Label(
                id="standard",
                query="question",
                answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some", id="123"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
                filters={"from_filter": 123},
                meta={"from_meta": ["123"]},
            ),
            # different answer in same doc
            Label(
                id="diff-answer-same-doc",
                query="question",
                answer=Answer(answer="answer2", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some", id="123"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
                filters={"from_filter": 123},
                meta={"from_meta": ["123"]},
            ),
            # answer in different doc
            Label(
                id="diff-answer-diff-doc",
                query="question",
                answer=Answer(answer="answer3", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some other", id="333"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
                filters={"from_filter": 333},
                meta={"from_meta": ["333"]},
            ),
            # 'no answer'
            Label(
                id="no-answer",
                query="question",
                answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
                document=Document(content="some", id="777"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
                filters={"from_filter": 777},
                meta={"from_meta": ["777"]},
            ),
            # different id, meta, same filters
            Label(
                id="5-888",
                query="question",
                answer=Answer(answer="answer5", offsets_in_document=[Span(start=12, end=18)]),
                document=Document(content="some", id="123"),
                is_correct_answer=True,
                is_correct_document=True,
                origin="gold-label",
                filters={"from_filter": 123},
                meta={"from_meta": ["888"]},
            ),
        ]

    @pytest.mark.unit
    def test_label_aggregation(self, standard_labels: List[Label]):
        multi_labels = aggregate_labels(standard_labels)
        assert len(multi_labels) == 1
        assert len(multi_labels[0].labels) == 6
        assert len(multi_labels[0].answers) == 4
        assert len(multi_labels[0].document_ids) == 4
        assert multi_labels[0].no_answer is False

    @pytest.mark.unit
    def test_label_aggregation_drop_negatives(self, standard_labels: List[Label]):
        multi_labels = aggregate_labels(standard_labels, drop_negative_labels=True)
        assert len(multi_labels) == 1
        assert len(multi_labels[0].labels) == 5
        assert len(multi_labels[0].answers) == 3
        assert "5-negative" not in [l.id for l in multi_labels[0].labels]
        assert len(multi_labels[0].document_ids) == 3
        assert multi_labels[0].no_answer is False

    @pytest.mark.unit
    def test_label_aggregation_drop_no_answers(self, standard_labels: List[Label]):
        multi_labels = aggregate_labels(standard_labels, drop_no_answers=True)
        assert len(multi_labels) == 1
        assert len(multi_labels[0].labels) == 4
        assert len(multi_labels[0].answers) == 4
        assert len(multi_labels[0].document_ids) == 4
        assert multi_labels[0].no_answer is False

    @pytest.mark.unit
    def test_label_aggregation_drop_negatives_and_no_answers(self, standard_labels: List[Label]):
        multi_labels = aggregate_labels(standard_labels, drop_negative_labels=True, drop_no_answers=True)
        assert len(multi_labels) == 1
        assert len(multi_labels[0].labels) == 3
        assert len(multi_labels[0].answers) == 3
        assert len(multi_labels[0].document_ids) == 3
        assert multi_labels[0].no_answer is False

    @pytest.mark.unit
    def test_label_aggregation_closed_domain(self, standard_labels: List[Label]):
        multi_labels = aggregate_labels(standard_labels, add_closed_domain_filter=True)
        assert len(multi_labels) == 3
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [4, 1, 1]
        assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)
        assert multi_labels[0].no_answer is False
        assert multi_labels[1].no_answer is False
        assert multi_labels[2].no_answer is True
        for ml in multi_labels:
            assert "_id" in ml.filters

    @pytest.mark.unit
    def test_label_aggregation_closed_domain_drop_negatives(self, standard_labels: List[Label]):
        multi_labels = aggregate_labels(standard_labels, add_closed_domain_filter=True, drop_negative_labels=True)
        assert len(multi_labels) == 3
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [3, 1, 1]
        assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)
        assert multi_labels[0].no_answer is False
        assert multi_labels[1].no_answer is False
        assert multi_labels[2].no_answer is True
        for ml in multi_labels:
            assert "_id" in ml.filters

    @pytest.mark.unit
    def test_aggregate_labels_filter_aggregations_with_no_sequence_values(self, filter_meta_labels: List[Label]):
        multi_labels = aggregate_labels(filter_meta_labels)
        assert len(multi_labels) == 3
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [3, 1, 1]
        assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)
        for multi_label in multi_labels:
            for l in multi_label.labels:
                assert "from_filter" in l.filters
                assert multi_label.filters == l.filters

    @pytest.mark.unit
    def test_aggregate_labels_filter_aggregations_with_string_values(self, filter_meta_labels: List[Label]):
        for label in filter_meta_labels:
            label.filters["from_filter"] = str(label.filters["from_filter"])

        multi_labels = aggregate_labels(filter_meta_labels)
        assert len(multi_labels) == 3
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [3, 1, 1]
        assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)
        for multi_label in multi_labels:
            for l in multi_label.labels:
                assert "from_filter" in l.filters
                assert multi_label.filters == l.filters

    @pytest.mark.unit
    def test_aggregate_labels_filter_aggregations_with_list_values(self, filter_meta_labels: List[Label]):
        for label in filter_meta_labels:
            label.filters["from_filter"] = [label.filters["from_filter"], "some_other_value"]

        multi_labels = aggregate_labels(filter_meta_labels)
        assert len(multi_labels) == 3
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [3, 1, 1]
        assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)
        for multi_label in multi_labels:
            for l in multi_label.labels:
                assert "from_filter" in l.filters
                assert multi_label.filters == l.filters

    @pytest.mark.unit
    def test_aggregate_labels_filter_aggregations_with_no_sequence_values_closed_domain(
        self, filter_meta_labels: List[Label]
    ):
        multi_labels = aggregate_labels(filter_meta_labels, add_closed_domain_filter=True)
        assert len(multi_labels) == 3
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [3, 1, 1]
        assert len(multi_labels[0].answers) == len(multi_labels[0].document_ids)
        for multi_label in multi_labels:
            for l in multi_label.labels:
                assert "from_filter" in l.filters
                assert "_id" in l.filters
                assert multi_label.filters == l.filters

    @pytest.mark.unit
    def test_aggregate_labels_meta_aggregations(self, filter_meta_labels: List[Label]):
        multi_labels = aggregate_labels(filter_meta_labels, add_meta_filters="from_meta")
        assert len(multi_labels) == 4
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [2, 1, 1, 1]
        for multi_label in multi_labels:
            for l in multi_label.labels:
                assert "from_filter" in l.filters
                assert l.filters["from_meta"] == l.meta["from_meta"]
                assert multi_label.filters == l.filters

    @pytest.mark.unit
    def test_aggregate_labels_meta_aggregations_closed_domain(self, filter_meta_labels: List[Label]):
        multi_labels = aggregate_labels(filter_meta_labels, add_closed_domain_filter=True, add_meta_filters="from_meta")
        assert len(multi_labels) == 4
        label_counts = [len(ml.labels) for ml in multi_labels]
        assert label_counts == [2, 1, 1, 1]
        for multi_label in multi_labels:
            for l in multi_label.labels:
                assert "from_filter" in l.filters
                assert l.filters["from_meta"] == l.meta["from_meta"]
                assert "_id" in l.filters
                assert multi_label.filters == l.filters


@pytest.mark.unit
def test_print_answers_run():
    with mock.patch("pprint.PrettyPrinter.pprint") as pprint:
        query_string = "Who is the father of Arya Stark?"
        run_result = {
            "query": query_string,
            "answers": [Answer(answer="Eddard", context="Eddard"), Answer(answer="Ned", context="Eddard")],
        }

        print_answers(run_result, details="minimum")

        expected_pprint_string = f"Query: {query_string}"
        pprint.assert_any_call(expected_pprint_string)

        expected_pprint_answers = [
            {"answer": answer.answer, "context": answer.context}  # filtered fields for minimum
            for answer in run_result["answers"]
        ]
        pprint.assert_any_call(expected_pprint_answers)


@pytest.mark.unit
def test_print_answers_run_batch():
    with mock.patch("pprint.PrettyPrinter.pprint") as pprint:
        queries = ["Who is the father of Arya Stark?", "Who is the sister of Arya Stark?"]
        answers = [
            [Answer(answer="Eddard", context="Eddard"), Answer(answer="Ned", context="Eddard")],
            [Answer(answer="Sansa", context="Sansa")],
        ]
        run_batch_result = {"queries": queries, "answers": answers}

        print_answers(run_batch_result, details="minimum")

        for query in queries:
            expected_pprint_string = f"Query: {query}"
            pprint.assert_any_call(expected_pprint_string)
        for answer_list in answers:
            expected_pprint_answers = [
                {"answer": answer.answer, "context": answer.context}  # filtered fields for minimum
                for answer in answer_list
            ]
            pprint.assert_any_call(expected_pprint_answers)
