import logging
import numpy as np
import pytest
import pandas as pd
from pathlib import Path

import responses
from responses import matchers
from haystack.utils.deepsetcloud import DeepsetCloud

from haystack.utils.preprocessing import convert_files_to_docs, tika_convert_files_to_docs
from haystack.utils.cleaning import clean_wiki_text
from haystack.utils.augment_squad import augment_squad
from haystack.utils.squad_data import SquadData
from haystack.utils.context_matching import calculate_context_similarity, match_context, match_contexts

from ..conftest import DC_API_ENDPOINT, DC_API_KEY, MOCK_DC, SAMPLES_PATH, deepset_cloud_fixture

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


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_upload_file_to_deepset_cloud(caplog):
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/files",
            json={"file_id": "abc"},
            status=200,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/files",
            json={"file_id": "def"},
            status=200,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/files",
            json={"file_id": "def"},
            status=200,
        )

    client = DeepsetCloud.get_file_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    file_paths = [
        SAMPLES_PATH / "docx/sample_docx.docx",
        SAMPLES_PATH / "pdf/sample_pdf_1.pdf",
        SAMPLES_PATH / "docs/doc_1.txt",
    ]
    metas = [{"file_id": "sample_docx.docx"}, {"file_id": "sample_pdf_1.pdf"}, {"file_id": "doc_1.txt"}]
    with caplog.at_level(logging.INFO):
        client.upload_files(file_paths=file_paths, metas=metas)
        assert f"Successfully uploaded {len(file_paths)} files." in caplog.text


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_upload_file_to_deepset_cloud_file_fails(caplog):
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/files",
            json={"file_id": "abc"},
            status=200,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/files",
            json={"error": "my-error"},
            status=500,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/files",
            json={"file_id": "def"},
            status=200,
        )

    client = DeepsetCloud.get_file_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    file_paths = [
        SAMPLES_PATH / "docx/sample_docx.docx",
        SAMPLES_PATH / "pdf/sample_pdf_1.pdf",
        SAMPLES_PATH / "docs/doc_1.txt",
    ]
    metas = [{"file_id": "sample_docx.docx"}, {"file_id": "sample_pdf_1.pdf"}, {"file_id": "doc_1.txt"}]
    with caplog.at_level(logging.INFO):
        client.upload_files(file_paths=file_paths, metas=metas)
        assert f"Successfully uploaded 2 files." in caplog.text
        assert f"Error uploading file" in caplog.text
        assert f"my-error" in caplog.text


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_delete_file_to_deepset_cloud():
    if MOCK_DC:
        responses.add(method=responses.DELETE, url=f"{DC_API_ENDPOINT}/workspaces/default/files/abc", status=200)

    client = DeepsetCloud.get_file_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    client.delete_file(file_id="abc")


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_list_files_on_deepset_cloud():
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/files",
            json={
                "data": [
                    {
                        "characters": -1,
                        "created_at": "2022-05-19T15:40:07.538162+00:00",
                        "file_id": "b6cdd48b-3db5-488b-a44d-4240c12a96d5",
                        "languages": [],
                        "meta": {},
                        "name": "sample_pdf_1.pdf",
                        "params": {"id_hash_keys": ["content", "meta"]},
                        "size": 44524,
                        "url": "/api/v1/workspaces/e282219f-19b2-41ff-927e-bda4e6e67418/files/b6cdd48b-3db5-488b-a44d-4240c12a96d5",
                    },
                    {
                        "characters": -1,
                        "created_at": "2022-05-23T12:39:53.393716+00:00",
                        "file_id": "51e9c2af-5676-453d-9b71-db9a560ae266",
                        "languages": [],
                        "meta": {"file_id": "sample_pdf_2.pdf"},
                        "name": "sample_pdf_2.pdf",
                        "params": {"id_hash_keys": ["content", "meta"]},
                        "size": 26093,
                        "url": "/api/v1/workspaces/e282219f-19b2-41ff-927e-bda4e6e67418/files/51e9c2af-5676-453d-9b71-db9a560ae266",
                    },
                ],
                "has_more": False,
                "total": 2,
            },
            status=200,
        )

    client = DeepsetCloud.get_file_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    files = [f for f in client.list_files()]
    assert len(files) == 2
    assert files[0]["name"] == "sample_pdf_1.pdf"
    assert files[1]["name"] == "sample_pdf_2.pdf"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_create_eval_run():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs",
            json={"data": {"eval_run_name": "my-eval-run-1"}},
            status=200,
            match=[
                matchers.json_params_matcher(
                    {
                        "name": "my-eval-run-1",
                        "pipeline_name": "my-pipeline-1",
                        "evaluation_set_name": "my-eval-set-1",
                        "eval_mode": 0,
                        "comment": "this is my first run",
                        "debug": False,
                        "tags": ["my-experiment-1"],
                    }
                )
            ],
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs",
            json={
                "data": [
                    {
                        "created_at": "2022-05-24T12:13:16.445857+00:00",
                        "eval_mode": 0,
                        "eval_run_id": "17875c63-7c07-42d8-bb01-4fcd95ce113c",
                        "name": "my-eval-run-1",
                        "comment": "this is my first run",
                        "tags": ["my-experiment-1"],
                        "eval_run_labels": [],
                        "logs": {},
                        "metrics": {
                            "integrated_exact_match": None,
                            "integrated_f1": None,
                            "integrated_sas": None,
                            "isolated_exact_match": None,
                            "isolated_f1": None,
                            "isolated_sas": None,
                            "mean_average_precision": None,
                            "mean_reciprocal_rank": None,
                            "normal_discounted_cummulative_gain": None,
                            "precision": None,
                            "recall_multi_hit": None,
                            "recall_single_hit": None,
                        },
                        "parameters": {
                            "debug": False,
                            "eval_mode": 0,
                            "evaluation_set_name": "my-eval-set-1",
                            "pipeline_name": "my-pipeline-1",
                        },
                        "status": 1,
                    }
                ],
                "has_more": False,
                "total": 1,
            },
            status=200,
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1",
            json={
                "created_at": "2022-05-24T12:13:16.445857+00:00",
                "eval_mode": 0,
                "eval_run_id": "17875c63-7c07-42d8-bb01-4fcd95ce113c",
                "name": "my-eval-run-1",
                "comment": "this is my first run",
                "tags": ["my-experiment-1"],
                "eval_run_labels": [],
                "logs": {},
                "metrics": {
                    "integrated_exact_match": None,
                    "integrated_f1": None,
                    "integrated_sas": None,
                    "isolated_exact_match": None,
                    "isolated_f1": None,
                    "isolated_sas": None,
                    "mean_average_precision": None,
                    "mean_reciprocal_rank": None,
                    "normal_discounted_cummulative_gain": None,
                    "precision": None,
                    "recall_multi_hit": None,
                    "recall_single_hit": None,
                },
                "parameters": {
                    "debug": False,
                    "eval_mode": 0,
                    "evaluation_set_name": "my-eval-set-1",
                    "pipeline_name": "my-pipeline-1",
                },
                "status": 1,
            },
            status=200,
        )

    client = DeepsetCloud.get_eval_run_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    client.create_eval_run(
        eval_run_name="my-eval-run-1",
        pipeline_config_name="my-pipeline-1",
        evaluation_set="my-eval-set-1",
        eval_mode="integrated",
        comment="this is my first run",
        tags=["my-experiment-1"],
    )

    runs = client.get_eval_runs()
    assert len(runs) == 1
    assert runs[0]["name"] == "my-eval-run-1"
    assert runs[0]["tags"] == ["my-experiment-1"]
    assert runs[0]["comment"] == "this is my first run"
    assert runs[0]["parameters"]["pipeline_name"] == "my-pipeline-1"
    assert runs[0]["parameters"]["evaluation_set_name"] == "my-eval-set-1"

    run = client.get_eval_run("my-eval-run-1")
    assert run["name"] == "my-eval-run-1"
    assert run["tags"] == ["my-experiment-1"]
    assert run["comment"] == "this is my first run"
    assert run["parameters"]["pipeline_name"] == "my-pipeline-1"
    assert run["parameters"]["evaluation_set_name"] == "my-eval-set-1"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_update_eval_run():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs",
            json={"data": {"eval_run_name": "my-eval-run-1"}},
            status=200,
            match=[
                matchers.json_params_matcher(
                    {
                        "name": "my-eval-run-1",
                        "pipeline_name": "my-pipeline-1",
                        "evaluation_set_name": "my-eval-set-1",
                        "eval_mode": 0,
                        "comment": "this is my first run",
                        "debug": False,
                        "tags": ["my-experiment-1"],
                    }
                )
            ],
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1",
            json={
                "created_at": "2022-05-24T12:13:16.445857+00:00",
                "eval_mode": 0,
                "eval_run_id": "17875c63-7c07-42d8-bb01-4fcd95ce113c",
                "name": "my-eval-run-1",
                "comment": "this is my first run",
                "tags": ["my-experiment-1"],
                "eval_run_labels": [],
                "logs": {},
                "metrics": {
                    "integrated_exact_match": None,
                    "integrated_f1": None,
                    "integrated_sas": None,
                    "isolated_exact_match": None,
                    "isolated_f1": None,
                    "isolated_sas": None,
                    "mean_average_precision": None,
                    "mean_reciprocal_rank": None,
                    "normal_discounted_cummulative_gain": None,
                    "precision": None,
                    "recall_multi_hit": None,
                    "recall_single_hit": None,
                },
                "parameters": {
                    "debug": False,
                    "eval_mode": 0,
                    "evaluation_set_name": "my-eval-set-1",
                    "pipeline_name": "my-pipeline-1",
                },
                "status": "CREATED",
            },
            status=200,
        )

        responses.add(
            method=responses.PATCH,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1",
            json={"data": {"eval_run_name": "my-eval-run-1"}},
            status=200,
            match=[
                matchers.json_params_matcher(
                    {"pipeline_name": "my-pipeline-2", "comment": "this is my first run with second pipeline"}
                )
            ],
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1",
            json={
                "created_at": "2022-05-24T12:13:16.445857+00:00",
                "eval_mode": 0,
                "eval_run_id": "17875c63-7c07-42d8-bb01-4fcd95ce113c",
                "name": "my-eval-run-1",
                "comment": "this is my first run with second pipeline",
                "tags": ["my-experiment-1"],
                "eval_run_labels": [],
                "logs": {},
                "metrics": {
                    "integrated_exact_match": None,
                    "integrated_f1": None,
                    "integrated_sas": None,
                    "isolated_exact_match": None,
                    "isolated_f1": None,
                    "isolated_sas": None,
                    "mean_average_precision": None,
                    "mean_reciprocal_rank": None,
                    "normal_discounted_cummulative_gain": None,
                    "precision": None,
                    "recall_multi_hit": None,
                    "recall_single_hit": None,
                },
                "parameters": {
                    "debug": False,
                    "eval_mode": 0,
                    "evaluation_set_name": "my-eval-set-1",
                    "pipeline_name": "my-pipeline-2",
                },
                "status": "CREATED",
            },
            status=200,
        )

    client = DeepsetCloud.get_eval_run_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    client.create_eval_run(
        eval_run_name="my-eval-run-1",
        pipeline_config_name="my-pipeline-1",
        evaluation_set="my-eval-set-1",
        eval_mode="integrated",
        comment="this is my first run",
        tags=["my-experiment-1"],
    )

    run = client.get_eval_run("my-eval-run-1")
    assert run["name"] == "my-eval-run-1"
    assert run["tags"] == ["my-experiment-1"]
    assert run["comment"] == "this is my first run"
    assert run["parameters"]["pipeline_name"] == "my-pipeline-1"
    assert run["parameters"]["evaluation_set_name"] == "my-eval-set-1"

    client.update_eval_run(
        eval_run_name="my-eval-run-1",
        pipeline_config_name="my-pipeline-2",
        comment="this is my first run with second pipeline",
    )

    run = client.get_eval_run("my-eval-run-1")
    assert run["name"] == "my-eval-run-1"
    assert run["tags"] == ["my-experiment-1"]
    assert run["comment"] == "this is my first run with second pipeline"
    assert run["parameters"]["pipeline_name"] == "my-pipeline-2"
    assert run["parameters"]["evaluation_set_name"] == "my-eval-set-1"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_start_eval_run():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs",
            json={"data": {"eval_run_name": "my-eval-run-1"}},
            status=200,
            match=[
                matchers.json_params_matcher(
                    {
                        "name": "my-eval-run-1",
                        "pipeline_name": "my-pipeline-1",
                        "evaluation_set_name": "my-eval-set-1",
                        "eval_mode": 0,
                        "comment": "this is my first run",
                        "debug": False,
                        "tags": ["my-experiment-1"],
                    }
                )
            ],
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1",
            json={
                "created_at": "2022-05-24T12:13:16.445857+00:00",
                "eval_mode": 0,
                "eval_run_id": "17875c63-7c07-42d8-bb01-4fcd95ce113c",
                "name": "my-eval-run-1",
                "comment": "this is my first run",
                "tags": ["my-experiment-1"],
                "eval_run_labels": [],
                "logs": {},
                "metrics": {
                    "integrated_exact_match": None,
                    "integrated_f1": None,
                    "integrated_sas": None,
                    "isolated_exact_match": None,
                    "isolated_f1": None,
                    "isolated_sas": None,
                    "mean_average_precision": None,
                    "mean_reciprocal_rank": None,
                    "normal_discounted_cummulative_gain": None,
                    "precision": None,
                    "recall_multi_hit": None,
                    "recall_single_hit": None,
                },
                "parameters": {
                    "debug": False,
                    "eval_mode": 0,
                    "evaluation_set_name": "my-eval-set-1",
                    "pipeline_name": "my-pipeline-1",
                },
                "status": "CREATED",
            },
            status=200,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1/start",
            json={},
            status=200,
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1",
            json={
                "created_at": "2022-05-24T12:13:16.445857+00:00",
                "eval_mode": 0,
                "eval_run_id": "17875c63-7c07-42d8-bb01-4fcd95ce113c",
                "name": "my-eval-run-1",
                "comment": "this is my first run",
                "tags": ["my-experiment-1"],
                "eval_run_labels": [],
                "logs": {},
                "metrics": {
                    "integrated_exact_match": None,
                    "integrated_f1": None,
                    "integrated_sas": None,
                    "isolated_exact_match": None,
                    "isolated_f1": None,
                    "isolated_sas": None,
                    "mean_average_precision": None,
                    "mean_reciprocal_rank": None,
                    "normal_discounted_cummulative_gain": None,
                    "precision": None,
                    "recall_multi_hit": None,
                    "recall_single_hit": None,
                },
                "parameters": {
                    "debug": False,
                    "eval_mode": 0,
                    "evaluation_set_name": "my-eval-set-1",
                    "pipeline_name": "my-pipeline-1",
                },
                "status": "STARTED",
            },
            status=200,
        )

    client = DeepsetCloud.get_eval_run_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    client.create_eval_run(
        eval_run_name="my-eval-run-1",
        pipeline_config_name="my-pipeline-1",
        evaluation_set="my-eval-set-1",
        eval_mode="integrated",
        comment="this is my first run",
        tags=["my-experiment-1"],
    )

    run = client.get_eval_run("my-eval-run-1")
    assert run["name"] == "my-eval-run-1"
    assert run["tags"] == ["my-experiment-1"]
    assert run["comment"] == "this is my first run"
    assert run["parameters"]["pipeline_name"] == "my-pipeline-1"
    assert run["parameters"]["evaluation_set_name"] == "my-eval-set-1"
    assert run["status"] == "CREATED"

    client.start_eval_run(eval_run_name="my-eval-run-1")

    run = client.get_eval_run("my-eval-run-1")
    assert run["name"] == "my-eval-run-1"
    assert run["tags"] == ["my-experiment-1"]
    assert run["comment"] == "this is my first run"
    assert run["parameters"]["pipeline_name"] == "my-pipeline-1"
    assert run["parameters"]["evaluation_set_name"] == "my-eval-set-1"
    assert run["status"] == "STARTED"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_delete_eval_run():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs",
            json={"data": {"eval_run_name": "my-eval-run-1"}},
            status=200,
            match=[
                matchers.json_params_matcher(
                    {
                        "name": "my-eval-run-1",
                        "pipeline_name": "my-pipeline-1",
                        "evaluation_set_name": "my-eval-set-1",
                        "eval_mode": 0,
                        "comment": "this is my first run",
                        "debug": False,
                        "tags": ["my-experiment-1"],
                    }
                )
            ],
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs",
            json={
                "data": [
                    {
                        "created_at": "2022-05-24T12:13:16.445857+00:00",
                        "eval_mode": 0,
                        "eval_run_id": "17875c63-7c07-42d8-bb01-4fcd95ce113c",
                        "name": "my-eval-run-1",
                        "comment": "this is my first run",
                        "tags": ["my-experiment-1"],
                        "eval_run_labels": [],
                        "logs": {},
                        "metrics": {
                            "integrated_exact_match": None,
                            "integrated_f1": None,
                            "integrated_sas": None,
                            "isolated_exact_match": None,
                            "isolated_f1": None,
                            "isolated_sas": None,
                            "mean_average_precision": None,
                            "mean_reciprocal_rank": None,
                            "normal_discounted_cummulative_gain": None,
                            "precision": None,
                            "recall_multi_hit": None,
                            "recall_single_hit": None,
                        },
                        "parameters": {
                            "debug": False,
                            "eval_mode": 0,
                            "evaluation_set_name": "my-eval-set-1",
                            "pipeline_name": "my-pipeline-1",
                        },
                        "status": 1,
                    }
                ],
                "has_more": False,
                "total": 1,
            },
            status=200,
        )

        responses.add(
            method=responses.DELETE, url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs/my-eval-run-1", status=204
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/eval_runs",
            json={"data": [], "has_more": False, "total": 0},
            status=200,
        )

    client = DeepsetCloud.get_eval_run_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    client.create_eval_run(
        eval_run_name="my-eval-run-1",
        pipeline_config_name="my-pipeline-1",
        evaluation_set="my-eval-set-1",
        eval_mode="integrated",
        comment="this is my first run",
        tags=["my-experiment-1"],
    )

    runs = client.get_eval_runs()
    assert len(runs) == 1

    run = client.delete_eval_run("my-eval-run-1")

    runs = client.get_eval_runs()
    assert len(runs) == 0


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_upload_eval_set(caplog):
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets/import",
            json={"evaluation_set_id": "c2d06025-2c00-43b5-8f73-b81b12e63afc"},
            status=200,
        )

    client = DeepsetCloud.get_evaluation_set_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    with caplog.at_level(logging.INFO):
        client.upload_evaluation_set(file_path=SAMPLES_PATH / "dc/matching_test_1.csv")
        assert f"Successfully uploaded evaluation set file" in caplog.text
        assert f"You can access it now under evaluation set 'matching_test_1.csv'." in caplog.text


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_upload_existing_eval_set(caplog):
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/evaluation_sets/import",
            json={"errors": ["Evaluation set with the same name already exists."]},
            status=409,
        )

    client = DeepsetCloud.get_evaluation_set_client(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    with caplog.at_level(logging.INFO):
        client.upload_evaluation_set(file_path=SAMPLES_PATH / "dc/matching_test_1.csv")
        assert f"Successfully uploaded evaluation set file" not in caplog.text
        assert f"You can access it now under evaluation set 'matching_test_1.csv'." not in caplog.text
        assert "Evaluation set with the same name already exists." in caplog.text
