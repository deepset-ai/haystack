from collections import namedtuple
from typing import Generator, Tuple, List
import re
from rapidfuzz import fuzz
from multiprocessing import Pool
from tqdm import tqdm


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
    candidates: Generator[Tuple[object, str], None, None],
    threshold: float = 60.0,
    show_progress: bool = False,
    num_processes: int = None,
    chunksize: int = None,
) -> List[Tuple[object, float]]:
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
    CandidateScore = namedtuple("CandidateScore", ["candidate_id", "score"])

    def score_candidate(candidate: Tuple[object, str]):
        candidate_id, candidate_text = candidate
        score = calculate_context_similarity(context, candidate_text)
        return CandidateScore(candidate_id, score)

    with Pool(processes=num_processes) as pool:
        candidate_scores = pool.imap_unordered(score_candidate, candidates, chunksize=chunksize)
        if show_progress:
            candidate_scores = tqdm(candidate_scores)

        matches = (candidate for candidate in candidate_scores if candidate.score > threshold)
        sorted_matches = sorted(matches, key=lambda candidate: candidate.score, reverse=True)
        match_list = list(sorted_matches)

        return match_list
