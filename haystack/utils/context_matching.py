from typing import Generator, Iterable, Optional, Tuple, List, Union

import re
from itertools import groupby
from multiprocessing.pool import Pool
from collections import namedtuple

from rapidfuzz import fuzz
from tqdm.auto import tqdm


_CandidateScore = namedtuple("_CandidateScore", ["context_id", "candidate_id", "score"])


def _score_candidate(args: Tuple[Union[str, Tuple[object, str]], Tuple[object, str], int, bool]):
    context, candidate, min_length, boost_split_overlaps = args
    candidate_id, candidate_text = candidate
    context_id, context_text = (None, context) if isinstance(context, str) else context
    score = calculate_context_similarity(
        context=context_text, candidate=candidate_text, min_length=min_length, boost_split_overlaps=boost_split_overlaps
    )
    return _CandidateScore(context_id=context_id, candidate_id=candidate_id, score=score)


def normalize_white_space_and_case(str: str) -> str:
    return re.sub(r"\s+", " ", str).lower().strip()


def _no_processor(str: str) -> str:
    return str


def calculate_context_similarity(
    context: str, candidate: str, min_length: int = 100, boost_split_overlaps: bool = True
) -> float:
    """
    Calculates the text similarity score of context and candidate.
    The score's value ranges between 0.0 and 100.0.

    :param context: The context to match.
    :param candidate: The candidate to match the context.
    :param min_length: The minimum string length context and candidate need to have in order to be scored.
                       Returns 0.0 otherwise.
    :param boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries
                                 we cut the context on the same side, recalculate the score and take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
    """
    # we need to handle short contexts/contents (e.g single word)
    # as they produce high scores by matching if the chars of the word are contained in the other one
    # this has to be done after normalizing
    context = normalize_white_space_and_case(context)
    candidate = normalize_white_space_and_case(candidate)
    context_len = len(context)
    candidate_len = len(candidate)
    if candidate_len < min_length or context_len < min_length:
        return 0.0

    if context_len < candidate_len:
        shorter = context
        longer = candidate
        shorter_len = context_len
        longer_len = candidate_len
    else:
        shorter = candidate
        longer = context
        shorter_len = candidate_len
        longer_len = context_len

    score_alignment = fuzz.partial_ratio_alignment(shorter, longer, processor=_no_processor)
    score = score_alignment.score  # type: ignore [union-attr]

    # Special handling for split overlaps (e.g. [AB] <-> [BC]):
    # If we detect that the score is near a half match and the best fitting part of longer is at its boundaries
    # we cut the shorter on the same side, recalculate the score and take the mean of both.
    # Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total
    if boost_split_overlaps and 40 <= score < 65:
        cut_shorter_left = score_alignment.dest_start == 0  # type: ignore [union-attr]
        cut_shorter_right = score_alignment.dest_end == longer_len  # type: ignore [union-attr]
        cut_len = shorter_len // 2

        if cut_shorter_left:
            cut_score = fuzz.partial_ratio(shorter[cut_len:], longer, processor=_no_processor)
            if cut_score > score:
                score = (score + cut_score) / 2
        if cut_shorter_right:
            cut_score = fuzz.partial_ratio(shorter[:-cut_len], longer, processor=_no_processor)
            if cut_score > score:
                score = (score + cut_score) / 2

    return score


def match_context(
    context: str,
    candidates: Generator[Tuple[str, str], None, None],
    threshold: float = 65.0,
    show_progress: bool = False,
    num_processes: Optional[int] = None,
    chunksize: int = 1,
    min_length: int = 100,
    boost_split_overlaps: bool = True,
) -> List[Tuple[str, float]]:
    """
    Matches the context against multiple candidates. Candidates consist of a tuple of an id and its text.

    Returns a sorted list of the candidate ids and its scores filtered by the threshold in descending order.

    :param context: The context to match.
    :param candidates: The candidates to match the context.
                       A candidate consists of a tuple of candidate id and candidate text.
    :param threshold: Score threshold that candidates must surpass to be included into the result list.
    :param show_progress: Whether to show the progress of matching all candidates.
    :param num_processes: The number of processes to be used for matching in parallel.
    :param chunksize: The chunksize used during parallel processing.
                      If not specified chunksize is 1.
                      For very long iterables using a large value for chunksize can make the job complete much faster than using the default value of 1.
    :param min_length: The minimum string length context and candidate need to have in order to be scored.
                       Returns 0.0 otherwise.
    :param boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries
                                 we cut the context on the same side, recalculate the score and take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
    """
    pool: Optional[Pool] = None
    try:
        score_candidate_args = ((context, candidate, min_length, boost_split_overlaps) for candidate in candidates)
        if num_processes is None or num_processes > 1:
            pool = Pool(processes=num_processes)
            candidate_scores: Iterable = pool.imap_unordered(
                _score_candidate, score_candidate_args, chunksize=chunksize
            )
        else:
            candidate_scores = map(_score_candidate, score_candidate_args)

        if show_progress:
            candidate_scores = tqdm(candidate_scores)

        matches = (candidate for candidate in candidate_scores if candidate.score > threshold)
        sorted_matches = sorted(matches, key=lambda candidate: candidate.score, reverse=True)
        match_list = list((candidate_score.candidate_id, candidate_score.score) for candidate_score in sorted_matches)

        return match_list

    finally:
        if pool:
            pool.close()
            pool.join()


def match_contexts(
    contexts: List[str],
    candidates: Generator[Tuple[str, str], None, None],
    threshold: float = 65.0,
    show_progress: bool = False,
    num_processes: Optional[int] = None,
    chunksize: int = 1,
    min_length: int = 100,
    boost_split_overlaps: bool = True,
) -> List[List[Tuple[str, float]]]:
    """
    Matches the contexts against multiple candidates. Candidates consist of a tuple of an id and its string text.
    This method iterates over candidates only once.

    Returns for each context a sorted list of the candidate ids and its scores filtered by the threshold in descending order.

    :param contexts: The contexts to match.
    :param candidates: The candidates to match the context.
                       A candidate consists of a tuple of candidate id and candidate text.
    :param threshold: Score threshold that candidates must surpass to be included into the result list.
    :param show_progress: Whether to show the progress of matching all candidates.
    :param num_processes: The number of processes to be used for matching in parallel.
    :param chunksize: The chunksize used during parallel processing.
                      If not specified chunksize is 1.
                      For very long iterables using a large value for chunksize can make the job complete much faster than using the default value of 1.
    :param min_length: The minimum string length context and candidate need to have in order to be scored.
                              Returns 0.0 otherwise.
    :param boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries
                                 we cut the context on the same side, recalculate the score and take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
    """
    pool: Optional[Pool] = None
    try:
        score_candidate_args = (
            (context, candidate, min_length, boost_split_overlaps)
            for candidate in candidates
            for context in enumerate(contexts)
        )

        if num_processes is None or num_processes > 1:
            pool = Pool(processes=num_processes)
            candidate_scores: Iterable = pool.imap_unordered(
                _score_candidate, score_candidate_args, chunksize=chunksize
            )
        else:
            candidate_scores = map(_score_candidate, score_candidate_args)

        if show_progress:
            candidate_scores = tqdm(candidate_scores)

        match_lists: List[List[Tuple[str, float]]] = list()
        matches = (candidate for candidate in candidate_scores if candidate.score > threshold)
        group_sorted_matches = sorted(matches, key=lambda candidate: candidate.context_id)
        grouped_matches = groupby(group_sorted_matches, key=lambda candidate: candidate.context_id)
        for context_id, group in grouped_matches:
            sorted_group = sorted(group, key=lambda candidate: candidate.score, reverse=True)
            match_list = list((candiate_score.candidate_id, candiate_score.score) for candiate_score in sorted_group)
            match_lists.insert(context_id, match_list)

        return match_lists

    finally:
        if pool:
            pool.close()
            pool.join()
