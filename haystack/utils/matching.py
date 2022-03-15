from collections import namedtuple
from typing import Generator, Tuple, List, Union
import re
from rapidfuzz import fuzz
from multiprocessing import Pool
from tqdm import tqdm
from itertools import groupby


_CandidateScore = namedtuple("_CandidateScore", ["context_id", "candidate_id", "score"])


def _score_candidate(args: Tuple[Union[str, Tuple[object, str]], Tuple[object, str]]):
    context, candidate = args
    candidate_id, candidate_text = candidate
    context_id, context_text = (None, context) if isinstance(context, str) else context
    score = calculate_context_similarity(context_text, candidate_text)
    return _CandidateScore(context_id=context_id, candidate_id=candidate_id, score=score)


def normalize_white_space_and_case(str):
    return re.sub(r"\s+", " ", str).lower().strip()


def calculate_context_similarity(context: str, candidate: str) -> float:
    """
    Calculates the text similarity score of context and candidate.
    The score's value ranges between 0.0 and 100.0.

    :param context: The context to match.
    :param candidate: The candidate to match the context.
    """
    # we need to handle short contexts/contents (e.g single word)
    # as they produce high scores by matching if the chars of the word are contained in the other one
    n_context = len(context.split())
    n_content = len(candidate.split())
    if n_content < 25 or n_context < 25:
        return 0.0
    return fuzz.partial_ratio(context, candidate, processor=normalize_white_space_and_case)


def match_context(
    context: str,
    candidates: Generator[Tuple[str, str], None, None],
    threshold: float = 60.0,
    show_progress: bool = False,
    num_processes: int = None,
    chunksize: int = 1,
) -> List[Tuple[str, float]]:
    """
    Matches the context against multiple candidates. Candidates consist of a tuple of an id and its string representation.

    Returns a sorted list in descending order of the candidate ids and its scores that surpass the threshold.

    :param context: The context to match.
    :param candidates: The candidates to match the context.
                       A candidate consists of a tuple of candidate id and candidate text.
    :param threshold: Score threshold that candidates must surpass to be included into the result list.
    :param show_progress: Whether to show the progress of matching all candidates.
    :param num_processes: The number of processes to be used for matching in parallel.
    :param chunksize: The chunksize used during parallel processing.
                      If not specified chunksize is 1.
                      For very long iterables using a large value for chunksize can make the job complete much faster than using the default value of 1.
    """
    with Pool(processes=num_processes) as pool:
        score_candidate_args = ((context, candidate) for candidate in candidates)
        candidate_scores = pool.imap_unordered(_score_candidate, score_candidate_args, chunksize=chunksize)
        if show_progress:
            candidate_scores = tqdm(candidate_scores)

        matches = (candidate for candidate in candidate_scores if candidate.score > threshold)
        sorted_matches = sorted(matches, key=lambda candidate: candidate.score, reverse=True)
        match_list = list((candiate_score.candidate_id, candiate_score.score) for candiate_score in sorted_matches)

        return match_list


def match_contexts(
    contexts: List[str],
    candidates: Generator[Tuple[str, str], None, None],
    threshold: float = 60.0,
    show_progress: bool = False,
    num_processes: int = None,
    chunksize: int = 1,
) -> List[List[Tuple[str, float]]]:
    """
    Matches the contexts against multiple candidates. Candidates consist of a tuple of an id and its string representation.
    Candidates are passed only once.

    Returns a sorted list in descending order of the candidate ids and its scores that surpass the threshold.

    :param context: The context to match.
    :param candidates: The candidates to match the context.
                       A candidate consists of a tuple of candidate id and candidate text.
    :param threshold: Score threshold that candidates must surpass to be included into the result list.
    :param show_progress: Whether to show the progress of matching all candidates.
    :param num_processes: The number of processes to be used for matching in parallel.
    :param chunksize: The chunksize used during parallel processing.
                      If not specified chunksize is 1.
                      For very long iterables using a large value for chunksize can make the job complete much faster than using the default value of 1.
    """
    with Pool(processes=num_processes) as pool:
        score_candidate_args = ((context, candidate) for candidate in candidates for context in enumerate(contexts))
        candidate_scores = pool.imap_unordered(_score_candidate, score_candidate_args, chunksize=chunksize)
        if show_progress:
            candidate_scores = tqdm(candidate_scores)

        match_lists: List[List[Tuple[object, float]]] = list()
        matches = (candidate for candidate in candidate_scores if candidate.score > threshold)
        group_sorted_matches = sorted(matches, key=lambda candidate: candidate.context_id)
        grouped_matches = groupby(group_sorted_matches, key=lambda candidate: candidate.context_id)
        for context_id, group in grouped_matches:
            sorted_group = sorted(group, key=lambda candidate: candidate.score, reverse=True)
            match_list = list((candiate_score.candidate_id, candiate_score.score) for candiate_score in sorted_group)
            match_lists.insert(context_id, match_list)

        return match_lists
