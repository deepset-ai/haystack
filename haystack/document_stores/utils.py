import typing
from typing import Dict, List, Optional, Tuple, Union, Generator

import json
import logging
from datetime import datetime

from haystack.schema import Document, Label, Answer, Span
from haystack.nodes.preprocessor import PreProcessor

if typing.TYPE_CHECKING:
    # This results in a circular import if we don't use typing.TYPE_CHECKING
    from haystack.document_stores.base import BaseDocumentStore


logger = logging.getLogger(__name__)


def eval_data_from_json(
    filename: str,
    max_docs: Optional[Union[int, bool]] = None,
    preprocessor: Optional[PreProcessor] = None,
    open_domain: bool = False,
) -> Tuple[List[Document], List[Label]]:
    """
    Read Documents + Labels from a SQuAD-style file.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    :param filename: Path to file in SQuAD format
    :param max_docs: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
    :param open_domain: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.
    """
    docs: List[Document] = []
    labels = []
    problematic_ids = []

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
        if "title" not in data["data"][0]:
            logger.warning("No title information found for documents in QA file: %s", filename)

        for squad_document in data["data"]:
            if max_docs:
                if len(docs) > max_docs:
                    break
            # Extracting paragraphs and their labels from a SQuAD document dict
            cur_docs, cur_labels, cur_problematic_ids = _extract_docs_and_labels_from_dict(
                squad_document, preprocessor, open_domain
            )
            docs.extend(cur_docs)
            labels.extend(cur_labels)
            problematic_ids.extend(cur_problematic_ids)
    if len(problematic_ids) > 0:
        logger.warning(
            f"Could not convert an answer for {len(problematic_ids)} questions.\n"
            f"There were conversion errors for question ids: {problematic_ids}"
        )
    return docs, labels


def eval_data_from_jsonl(
    filename: str,
    batch_size: Optional[int] = None,
    max_docs: Optional[Union[int, bool]] = None,
    preprocessor: Optional[PreProcessor] = None,
    open_domain: bool = False,
) -> Generator[Tuple[List[Document], List[Label]], None, None]:
    """
    Read Documents + Labels from a SQuAD-style file in jsonl format, i.e. one document per line.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    This is a generator which will yield one tuple per iteration containing a list
    of batch_size documents and a list with the documents' labels.
    If batch_size is set to None, this method will yield all documents and labels.

    :param filename: Path to file in SQuAD format
    :param max_docs: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
    :param open_domain: Set this to True if your file is an open domain dataset where two different answers to the same question might be found in different contexts.
    """
    docs: List[Document] = []
    labels = []
    problematic_ids = []

    with open(filename, "r", encoding="utf-8") as file:
        for document in file:
            if max_docs:
                if len(docs) > max_docs:
                    break
            # Extracting paragraphs and their labels from a SQuAD document dict
            squad_document = json.loads(document)
            cur_docs, cur_labels, cur_problematic_ids = _extract_docs_and_labels_from_dict(
                squad_document, preprocessor, open_domain
            )
            docs.extend(cur_docs)
            labels.extend(cur_labels)
            problematic_ids.extend(cur_problematic_ids)

            if batch_size is not None:
                if len(docs) >= batch_size:
                    if len(problematic_ids) > 0:
                        logger.warning(
                            f"Could not convert an answer for {len(problematic_ids)} questions.\n"
                            f"There were conversion errors for question ids: {problematic_ids}"
                        )
                    yield docs, labels
                    docs = []
                    labels = []
                    problematic_ids = []

    yield docs, labels


def squad_json_to_jsonl(squad_file: str, output_file: str):
    """
    Converts a SQuAD-json-file into jsonl format with one document per line.

    :param squad_file: SQuAD-file in json format.
    :param output_file: Name of output file (SQuAD in jsonl format)
    """
    with open(squad_file, encoding="utf-8") as json_file, open(output_file, "w", encoding="utf-8") as jsonl_file:
        squad_json = json.load(json_file)

        for doc in squad_json["data"]:
            json.dump(doc, jsonl_file)
            jsonl_file.write("\n")


def _extract_docs_and_labels_from_dict(
    document_dict: Dict, preprocessor: Optional[PreProcessor] = None, open_domain: bool = False
):
    """
    Set open_domain to True if you are trying to load open_domain labels (i.e. labels without doc id or start idx)
    """
    docs = []
    labels = []
    problematic_ids = []

    # get all extra fields from document level (e.g. title)
    meta_doc = {k: v for k, v in document_dict.items() if k not in ("paragraphs", "title")}
    for paragraph in document_dict["paragraphs"]:
        ## Create Metadata
        cur_meta = {"name": document_dict.get("title", None)}
        # all other fields from paragraph level
        meta_paragraph = {k: v for k, v in paragraph.items() if k not in ("qas", "context")}
        cur_meta.update(meta_paragraph)
        # meta from parent document
        cur_meta.update(meta_doc)

        ## Create Document
        cur_full_doc = Document(content=paragraph["context"], meta=cur_meta)
        if preprocessor is not None:
            splits_docs = preprocessor.process(documents=[cur_full_doc])
            # we need to pull in _split_id into the document id for unique reference in labels
            splits: List[Document] = []
            offset = 0
            for d in splits_docs:
                id = f"{d.id}-{d.meta['_split_id']}"
                d.meta["_split_offset"] = offset
                offset += len(d.content)
                # offset correction based on splitting method
                if preprocessor.split_by == "word":
                    offset += 1
                elif preprocessor.split_by == "passage":
                    offset += 2
                else:
                    raise NotImplementedError
                mydoc = Document(content=d.content, id=id, meta=d.meta)
                splits.append(mydoc)
        else:
            splits = [cur_full_doc]
        docs.extend(splits)

        ## Assign Labels to corresponding documents
        for qa in paragraph["qas"]:
            if not qa.get("is_impossible", False):
                for answer in qa["answers"]:
                    ans = answer["text"]
                    # TODO The following block of code means that answer_start is never calculated
                    #  and cur_id is always None for open_domain
                    #  This can be rewritten so that this function could try to calculate offsets
                    #  and populate id in open_domain mode
                    if open_domain:
                        # TODO check with Branden why we want to treat open_domain here differently.
                        # Shouldn't this be something configured at eval time only?
                        cur_ans_start = answer.get("answer_start", 0)
                        # cur_id = '0'
                        label = Label(
                            query=qa["question"],
                            answer=Answer(answer=ans, type="extractive", score=0.0),
                            document=None,  # type: ignore
                            is_correct_answer=True,
                            is_correct_document=True,
                            origin="gold-label",
                        )
                        labels.append(label)
                    else:
                        ans_position = cur_full_doc.content[answer["answer_start"] : answer["answer_start"] + len(ans)]
                        if ans != ans_position:
                            # do not use answer
                            problematic_ids.append(qa.get("id", "missing"))
                            break
                        # find corresponding document or split
                        if len(splits) == 1:
                            # cur_id = splits[0].id
                            cur_ans_start = answer["answer_start"]
                            cur_doc = splits[0]
                        else:
                            for s in splits:
                                # If answer start offset is contained in passage we assign the label to that passage
                                if (answer["answer_start"] >= s.meta["_split_offset"]) and (
                                    answer["answer_start"] < (s.meta["_split_offset"] + len(s.content))
                                ):
                                    cur_doc = s
                                    cur_ans_start = answer["answer_start"] - s.meta["_split_offset"]
                                    # If a document is splitting an answer we add the whole answer text to the document
                                    if s.content[cur_ans_start : cur_ans_start + len(ans)] != ans:
                                        s.content = s.content[:cur_ans_start] + ans
                                    break
                        cur_answer = Answer(
                            answer=ans,
                            type="extractive",
                            score=0.0,
                            context=cur_doc.content,
                            offsets_in_document=[Span(start=cur_ans_start, end=cur_ans_start + len(ans))],
                            offsets_in_context=[Span(start=cur_ans_start, end=cur_ans_start + len(ans))],
                            document_id=cur_doc.id,
                        )
                        label = Label(
                            query=qa["question"],
                            answer=cur_answer,
                            document=cur_doc,
                            is_correct_answer=True,
                            is_correct_document=True,
                            origin="gold-label",
                        )
                        labels.append(label)
            else:
                # for no_answer we need to assign each split as not fitting to the question
                for s in splits:
                    label = Label(
                        query=qa["question"],
                        answer=Answer(
                            answer="",
                            type="extractive",
                            score=0.0,
                            offsets_in_document=[Span(start=0, end=0)],
                            offsets_in_context=[Span(start=0, end=0)],
                        ),
                        document=s,
                        is_correct_answer=True,
                        is_correct_document=True,
                        origin="gold-label",
                    )

                    labels.append(label)

    return docs, labels, problematic_ids


def convert_date_to_rfc3339(date: str) -> str:
    """
    Converts a date to RFC3339 format, as Weaviate requires dates to be in RFC3339 format including the time and
    timezone.

    If the provided date string does not contain a time and/or timezone, we use 00:00 as default time
    and UTC as default time zone.

    This method cannot be part of WeaviateDocumentStore, as this would result in a circular import between weaviate.py
    and filter_utils.py.
    """
    parsed_datetime = datetime.fromisoformat(date)
    if parsed_datetime.utcoffset() is None:
        converted_date = parsed_datetime.isoformat() + "Z"
    else:
        converted_date = parsed_datetime.isoformat()

    return converted_date
