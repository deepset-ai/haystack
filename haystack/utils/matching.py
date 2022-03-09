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
    """
    # we need to handle short contexts/contents (e.g single word) 
    # as they produce high scores by matching if the chars of the word are contained in the other one
    n_context = len(context.split())
    n_content = len(candidate.split())
    if n_content < 25 or n_context < 25:
        return 0.0
    return fuzz.partial_ratio(context, candidate, processor=normalize_white_space_and_case)

def match_context(context: str, candidates: Generator[Tuple[object, str], None, None], num_processes: int = None, threshold: float = 60.0) -> List[Tuple[object, float]]:
    """
    Matches the context against multiple candidates. Candidates consist of a tuple of an id and its string representation.

    Returns a sorted list in descending order of the candidate ids and its scores that surpass the threshold.
    """
    def process(candidate: Tuple[object, str]):
        candidate_id = candidate[0]
        candidate_str = candidate[1]
        score = calculate_context_similarity(context, candidate_str)
        return candidate_id, score

    with Pool(processes=num_processes) as pool:
        res = [(candidate_id, score) for candidate_id, score in tqdm(pool.imap(process, enumerate(candidates))) if score > threshold]
        return res
