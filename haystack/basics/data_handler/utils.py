import json
import logging
import os
import random
import tarfile
import tempfile
import uuid
from pathlib import Path

from haystack.basics.file_utils import http_get

logger = logging.getLogger(__name__)

DOWNSTREAM_TASK_MAP = {
    "squad20": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz",
    "covidqa": "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/covidqa.tar.gz",

}


def read_dpr_json(file, max_samples=None, proxies=None, num_hard_negatives=1, num_positives=1, shuffle_negatives=True, shuffle_positives=False):
    """
    Reads a Dense Passage Retrieval (DPR) data file in json format and returns a list of dictionaries.

    :param file: filename of DPR data in json format

    Returns:
        list of dictionaries: List[dict]
        each dictionary: {
                    "query": str -> query_text
                    "passages": List[dictionaries] -> [{"text": document_text, "title": xxx, "label": "positive", "external_id": abb123},
                                {"text": document_text, "title": xxx, "label": "hard_negative", "external_id": abb134},
                                ...]
                    }
        example:
                ["query": 'who sings does he love me with reba'
                "passages" : [{'title': 'Does He Love You',
                    'text': 'Does He Love You "Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba\'s album "Greatest Hits Volume Two". It is one of country music\'s several songs about a love triangle. "Does He Love You" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members',
                    'label': 'positive',
                    'external_id': '11828866'},
                    {'title': 'When the Nightingale Sings',
                    'text': "When the Nightingale Sings When The Nightingale Sings is a Middle English poem, author unknown, recorded in the British Library's Harley 2253 manuscript, verse 25. It is a love poem, extolling the beauty and lost love of an unknown maiden. When þe nyhtegale singes þe wodes waxen grene.<br> Lef ant gras ant blosme springes in aueryl y wene,<br> Ant love is to myn herte gon wiþ one spere so kene<br> Nyht ant day my blod hit drynkes myn herte deþ me tene. Ich have loved al þis er þat y may love namore,<br> Ich have siked moni syk lemmon for",
                    'label': 'hard_negative',
                    'external_id': '10891637'}]
                ]

    """
    # get remote dataset if needed
    if not (os.path.exists(file)):
        logger.info(f" Couldn't find {file} locally. Trying to download ...")
        _download_extract_downstream_data(file, proxies=proxies)

    if file.suffix.lower() == ".jsonl":
        dicts = []
        with open(file, encoding='utf-8') as f:
            for line in f:
                dicts.append(json.loads(line))
    else:
        dicts = json.load(open(file, encoding='utf-8'))

    if max_samples:
        dicts = random.sample(dicts, min(max_samples, len(dicts)))

    # convert DPR dictionary to standard dictionary
    query_json_keys = ["question", "questions", "query"]
    positive_context_json_keys = ["positive_contexts", "positive_ctxs", "positive_context", "positive_ctx"]
    hard_negative_json_keys = ["hard_negative_contexts", "hard_negative_ctxs", "hard_negative_context", "hard_negative_ctx"]
    standard_dicts = []
    for dict in dicts:
        sample = {}
        passages = []
        for key, val in dict.items():
            if key in query_json_keys:
                sample["query"] = val
            elif key in positive_context_json_keys:
                if shuffle_positives:
                    random.shuffle(val)
                for passage in val[:num_positives]:
                    passages.append({
                        "title": passage["title"],
                        "text": passage["text"],
                        "label": "positive",
                        "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8])
                        })
            elif key in hard_negative_json_keys:
                if shuffle_negatives:
                    random.shuffle(val)
                for passage in val[:num_hard_negatives]:
                    passages.append({
                        "title": passage["title"],
                        "text": passage["text"],
                        "label": "hard_negative",
                        "external_id": passage.get("passage_id", uuid.uuid4().hex.upper()[0:8])
                        })
        sample["passages"] = passages
        standard_dicts.append(sample)
    return standard_dicts


def read_squad_file(filename, proxies=None):
    """Read a SQuAD json file"""
    if not (os.path.exists(filename)):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies)
    with open(filename, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]
    return input_data


def write_squad_predictions(predictions, out_filename, predictions_filename=None):
    predictions_json = {}
    for x in predictions:
        for p in x["predictions"]:
            if p["answers"][0]["answer"] is not None:
                predictions_json[p["question_id"]] = p["answers"][0]["answer"]
            else:
                predictions_json[p["question_id"]] = "" #convert No answer = None to format understood by the SQuAD eval script

    if predictions_filename:
        dev_labels = {}
        temp = json.load(open(predictions_filename, "r"))
        for d in temp["data"]:
            for p in d["paragraphs"]:
                for q in p["qas"]:
                    if q.get("is_impossible",False):
                        dev_labels[q["id"]] = "is_impossible"
                    else:
                        dev_labels[q["id"]] = q["answers"][0]["text"]
        not_included = set(list(dev_labels.keys())) - set(list(predictions_json.keys()))
        if len(not_included) > 0:
            logger.info(f"There were missing predicitons for question ids: {list(not_included)}")
        for x in not_included:
            predictions_json[x] = ""

    # os.makedirs("model_output", exist_ok=True)
    # filepath = Path("model_output") / out_filename
    json.dump(predictions_json, open(out_filename, "w"))
    logger.info(f"Written Squad predictions to: {out_filename}")


def _download_extract_downstream_data(input_file, proxies=None):
    # download archive to temp dir and extract to correct position
    full_path = Path(os.path.realpath(input_file))
    directory = full_path.parent
    taskname = directory.stem
    datadir = directory.parent
    logger.info(
        "downloading and extracting file {} to dir {}".format(taskname, datadir)
    )
    if taskname not in DOWNSTREAM_TASK_MAP:
        logger.error("Cannot download {}. Unknown data source.".format(taskname))
    else:
        if os.name == "nt":  # make use of NamedTemporaryFile compatible with Windows
            delete_tmp_file = False
        else:
            delete_tmp_file = True
        with tempfile.NamedTemporaryFile(delete=delete_tmp_file) as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)
        # temp_file gets deleted here


def is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False
